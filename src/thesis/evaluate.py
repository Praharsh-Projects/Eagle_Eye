"""Retrieval and grounding evaluation for chunking strategy comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.thesis.retrieve import QueryFilters, ThesisRetriever, format_incident_aware_answer


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _match_port(meta: Dict[str, Any], expected_port: str) -> bool:
    exp = expected_port.upper().replace(" ", "")
    port = str(meta.get("port", "")).upper().replace(" ", "")
    locode = str(meta.get("locode", "")).upper().replace(" ", "")
    return exp in {port, locode} or exp in port


def _match_date(meta: Dict[str, Any], expected_date: str) -> bool:
    text = str(meta.get("date", ""))
    exp = expected_date[:10]
    return text[:10] == exp


def _match_anomaly(meta: Dict[str, Any], expected_anomaly: Any) -> bool:
    if expected_anomaly is None:
        return True
    return bool(meta.get("anomaly", False)) == bool(expected_anomaly)


def _query_hit(chunk_meta: Dict[str, Any], case: Dict[str, Any]) -> bool:
    expected_port = case.get("expected_port")
    expected_date = case.get("expected_date")
    expected_anomaly = case.get("expected_anomaly")

    if expected_port and not _match_port(chunk_meta, expected_port):
        return False
    if expected_date and not _match_date(chunk_meta, expected_date):
        return False
    if expected_anomaly is not None and not _match_anomaly(chunk_meta, expected_anomaly):
        return False
    return True


def _manual_relevance_heuristic(chunk_text: str, query: str) -> int:
    q_tokens = {t for t in re.findall(r"[a-zA-Z0-9]{4,}", query.lower())}
    c_tokens = {t for t in re.findall(r"[a-zA-Z0-9]{4,}", chunk_text.lower())}
    if not q_tokens:
        return 1
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    if overlap >= 0.5:
        return 2
    if overlap >= 0.2:
        return 1
    return 0


def _hallucination_flag(answer: str, evidence_text: str) -> bool:
    # Conservative lexical grounding check for capitalized location-like tokens.
    answer_tokens = {t for t in re.findall(r"\b[A-Z]{4,6}\b", answer)}
    evidence_tokens = {t for t in re.findall(r"\b[A-Z]{4,6}\b", evidence_text)}
    unknown = [t for t in answer_tokens if t not in evidence_tokens]
    return len(unknown) > 0


def _strategy_manifest(persist_dir: Path, strategy: str) -> Dict[str, Any]:
    manifest = persist_dir / f"manifest_{strategy.lower()}.json"
    if not manifest.exists():
        return {}
    with manifest.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_strategies(
    questions_path: Path,
    persist_dir: Path,
    strategies: List[str],
    embedding_model: str,
    top_k: int,
    out_dir: Path,
) -> Dict[str, Any]:
    cases = _read_jsonl(questions_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrievers = {
        s: ThesisRetriever(persist_dir=persist_dir, strategy=s, embedding_model=embedding_model)
        for s in strategies
    }

    summary: Dict[str, Any] = {"top_k": top_k, "strategies": {}}
    relevance_rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for strategy in strategies:
        retriever = retrievers[strategy]
        hits = 0
        total = 0
        total_latency = 0.0
        hallucinated_answers = 0

        for case in cases:
            total += 1
            query = case["query"]
            filters = QueryFilters(
                port=case.get("filter_port"),
                date_from=case.get("filter_date_from"),
                date_to=case.get("filter_date_to"),
                vessel_type=case.get("filter_vessel_type"),
                anomaly=case.get("filter_anomaly"),
            )

            retrieved = retriever.retrieve(query=query, top_k=top_k, filters=filters)
            total_latency += retrieved.latency_ms

            hit = any(_query_hit(chunk.metadata, case) for chunk in retrieved.chunks)
            if hit:
                hits += 1

            answer_payload = format_incident_aware_answer(query=query, retrieved=retrieved)
            evidence_text = "\n".join([c.text for c in retrieved.chunks])
            hallucinated = _hallucination_flag(answer_payload.get("answer", ""), evidence_text)
            hallucinated_answers += int(hallucinated)

            all_results.append(
                {
                    "strategy": strategy,
                    "query": query,
                    "hit": hit,
                    "latency_ms": retrieved.latency_ms,
                    "retrieved_count": len(retrieved.chunks),
                    "hallucinated": hallucinated,
                }
            )

            for rank, chunk in enumerate(retrieved.chunks, start=1):
                relevance_rows.append(
                    {
                        "strategy": strategy,
                        "query": query,
                        "rank": rank,
                        "chunk_id": chunk.chunk_id,
                        "port": chunk.metadata.get("port"),
                        "date": chunk.metadata.get("date"),
                        "distance": chunk.distance,
                        "auto_relevance_0_2": _manual_relevance_heuristic(chunk.text, query),
                        "manual_relevance_0_2": "",
                    }
                )

        hit_at_k = hits / max(1, total)
        mean_latency_ms = total_latency / max(1, total)
        hallucination_rate = hallucinated_answers / max(1, total)

        manifest = _strategy_manifest(persist_dir, strategy)
        summary["strategies"][strategy] = {
            "hit_at_k": round(hit_at_k, 4),
            "queries": total,
            "mean_latency_ms": round(mean_latency_ms, 3),
            "hallucination_rate": round(hallucination_rate, 4),
            "index_chunks": manifest.get("chunks"),
            "index_size_mb": manifest.get("persist_size_mb"),
            "embedding_model": manifest.get("embedding_model", embedding_model),
        }

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "per_query_results.csv", index=False)

    relevance_df = pd.DataFrame(relevance_rows)
    relevance_df.to_csv(out_dir / "manual_relevance_template.csv", index=False)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Charts for thesis figures.
    strat_labels = list(summary["strategies"].keys())
    hit_values = [summary["strategies"][s]["hit_at_k"] for s in strat_labels]
    lat_values = [summary["strategies"][s]["mean_latency_ms"] for s in strat_labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(strat_labels, hit_values, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(strat_labels)])
    ax.set_title("Hit@k by Chunking Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Hit@k")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "hit_at_k.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(strat_labels, lat_values, color=["#9467bd", "#17becf", "#8c564b"][: len(strat_labels)])
    ax.set_title("Mean Retrieval Latency by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Latency (ms)")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_ms.png", dpi=160)
    plt.close(fig)

    return summary


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate thesis retrieval strategies")
    parser.add_argument("--questions", default="evaluation/thesis/questions.jsonl")
    parser.add_argument("--persist_dir", default="data/thesis_chroma")
    parser.add_argument("--strategies", default="A,B,C", help="Comma-separated list, e.g. A,B")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_dir", default="evaluation/thesis/results")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    strategies = [s.strip().upper() for s in args.strategies.split(",") if s.strip()]
    summary = evaluate_strategies(
        questions_path=Path(args.questions),
        persist_dir=Path(args.persist_dir),
        strategies=strategies,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        out_dir=Path(args.out_dir),
    )
    print("Evaluation completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
