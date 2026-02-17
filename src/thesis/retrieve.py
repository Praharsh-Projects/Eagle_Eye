"""Retrieval and incident-aware answer formatting for thesis demo."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.runtime import import_chromadb


@dataclass
class QueryFilters:
    port: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    vessel_type: Optional[str] = None
    anomaly: Optional[bool] = None

    def normalized(self) -> "QueryFilters":
        return QueryFilters(
            port=self.port.strip().upper().replace(" ", "") if self.port else None,
            date_from=self.date_from,
            date_to=self.date_to,
            vessel_type=self.vessel_type.strip().lower() if self.vessel_type else None,
            anomaly=self.anomaly,
        )


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float]


@dataclass
class RetrievalOutput:
    strategy: str
    query: str
    chunks: List[RetrievedChunk]
    latency_ms: float


class ThesisRetriever:
    _MODEL_CACHE: Dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        persist_dir: str | Path = "data/thesis_chroma",
        strategy: str = "A",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.strategy = strategy.upper()
        self.embedding_model_name = embedding_model
        if embedding_model not in self._MODEL_CACHE:
            self._MODEL_CACHE[embedding_model] = SentenceTransformer(embedding_model)
        self.embedder = self._MODEL_CACHE[embedding_model]

        chromadb = import_chromadb()
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=f"thesis_{self.strategy.lower()}")

    def _embed_query(self, query: str) -> List[float]:
        vec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].tolist()

    @staticmethod
    def _build_where(filters: QueryFilters) -> Optional[Dict[str, Any]]:
        f = filters.normalized()
        clauses: List[Dict[str, Any]] = []
        if f.port:
            clauses.append({"port": {"$eq": f.port}})
        if f.vessel_type:
            clauses.append({"vessel_type": {"$eq": f.vessel_type}})
        if f.anomaly is not None:
            clauses.append({"anomaly": {"$eq": bool(f.anomaly)}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _in_date_range(date_value: Any, date_from: Optional[str], date_to: Optional[str]) -> bool:
        if date_value is None:
            return False
        text = str(date_value)[:10]
        if date_from and text < date_from:
            return False
        if date_to and text > date_to:
            return False
        return True

    @staticmethod
    def _match_filters(meta: Dict[str, Any], filters: QueryFilters) -> bool:
        f = filters.normalized()
        if f.port:
            port = str(meta.get("port", "")).upper().replace(" ", "")
            locode = str(meta.get("locode", "")).upper().replace(" ", "")
            if f.port not in {port, locode} and f.port not in str(meta.get("port", "")).upper():
                return False
        if f.vessel_type:
            if str(meta.get("vessel_type", "")).lower() != f.vessel_type:
                if f.vessel_type != "all":
                    return False
        if f.anomaly is not None:
            if bool(meta.get("anomaly", False)) != bool(f.anomaly):
                return False
        if f.date_from or f.date_to:
            if not ThesisRetriever._in_date_range(meta.get("date"), f.date_from, f.date_to):
                return False
        return True

    @staticmethod
    def _adaptive_n_results(top_k: int, available: int, filters: QueryFilters) -> int:
        f = filters.normalized()
        base = max(5, top_k * 3)
        if f.date_from or f.date_to:
            # Date filtering happens post-query, so request a larger candidate pool.
            if f.port or f.vessel_type or f.anomaly is not None:
                base = max(base, top_k * 80)
            else:
                base = max(base, top_k * 200)
        return min(base, available)

    @staticmethod
    def _cosine_distance(query_vec: List[float], emb: List[float]) -> float:
        q = np.asarray(query_vec, dtype=float)
        e = np.asarray(emb, dtype=float)
        qn = np.linalg.norm(q)
        en = np.linalg.norm(e)
        if qn == 0 or en == 0:
            return 1.0
        return float(1.0 - float(np.dot(q, e) / (qn * en)))

    def _fallback_rerank_by_date(
        self,
        query_vec: List[float],
        filters: QueryFilters,
        where: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[RetrievedChunk]:
        f = filters.normalized()
        if not (f.date_from or f.date_to):
            return []
        if where is None:
            return []

        try:
            candidates = self.collection.get(
                where=where,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            return []

        ids = candidates.get("ids")
        docs = candidates.get("documents")
        metas = candidates.get("metadatas")
        embs = candidates.get("embeddings")
        ids = ids if ids is not None else []
        docs = docs if docs is not None else []
        metas = metas if metas is not None else []
        embs = embs if embs is not None else []
        rescored: List[RetrievedChunk] = []

        for idx, cid in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            if not self._match_filters(meta, filters):
                continue
            emb = embs[idx] if idx < len(embs) else None
            if emb is None:
                continue
            dist = self._cosine_distance(query_vec, emb)
            rescored.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=docs[idx] if idx < len(docs) else "",
                    metadata=meta,
                    distance=dist,
                )
            )

        rescored.sort(key=lambda c: (c.distance if c.distance is not None else 999.0))
        return rescored[:top_k]

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[QueryFilters] = None) -> RetrievalOutput:
        start = time.perf_counter()
        filters = filters or QueryFilters()

        vec = self._embed_query(query)
        where = self._build_where(filters)

        available = self.collection.count()
        if available == 0:
            return RetrievalOutput(strategy=self.strategy, query=query, chunks=[], latency_ms=0.0)

        n_results = self._adaptive_n_results(top_k=top_k, available=available, filters=filters)
        try:
            result = self.collection.query(
                query_embeddings=[vec],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            result = self.collection.query(
                query_embeddings=[vec],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        chunks: List[RetrievedChunk] = []
        for idx, cid in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            if not self._match_filters(meta, filters):
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=docs[idx] if idx < len(docs) else "",
                    metadata=meta,
                    distance=dists[idx] if idx < len(dists) else None,
                )
            )
            if len(chunks) >= top_k:
                break

        if len(chunks) < top_k:
            fallback = self._fallback_rerank_by_date(
                query_vec=vec,
                filters=filters,
                where=where,
                top_k=top_k,
            )
            if fallback:
                chunks = fallback

        elapsed = (time.perf_counter() - start) * 1000.0
        return RetrievalOutput(strategy=self.strategy, query=query, chunks=chunks, latency_ms=elapsed)


def format_incident_aware_answer(query: str, retrieved: RetrievalOutput) -> Dict[str, Any]:
    if not retrieved.chunks:
        return {
            "answer": "I don't have sufficient data evidence for this query in the selected strategy and filters.",
            "evidence": [],
            "interpretation": "No relevant chunks were retrieved.",
            "suggested_check": "Broaden date/port filters or try another chunking strategy.",
        }

    evidence: List[str] = []
    anomaly_hits = 0
    for chunk in retrieved.chunks:
        meta = chunk.metadata
        if bool(meta.get("anomaly", False)):
            anomaly_hits += 1
        dist_text = f"{float(chunk.distance):.4f}" if chunk.distance is not None else "n/a"
        evidence.append(
            f"[{chunk.chunk_id}] port={meta.get('port','?')} date={meta.get('date','?')} "
            f"strategy={meta.get('strategy','?')} dist={dist_text}"
        )

    q = query.lower()
    if any(k in q for k in ["congestion", "busy", "pressure"]):
        interpretation = "Retrieved summaries indicate port-day traffic pressure patterns and dwell-related load indicators."
    elif any(k in q for k in ["anomaly", "suspicious", "jump", "incident"]):
        interpretation = "Retrieved events contain anomaly flags relevant to incident-aware monitoring."
    elif any(k in q for k in ["trend", "pattern", "forecast"]):
        interpretation = "Retrieved chunks provide historical patterns suitable for trend interpretation."
    else:
        interpretation = "Retrieved chunks are the closest historical evidence for the requested context."

    suggested_check = "Cross-check flagged vessels and compare with neighboring dates before operational action."
    if anomaly_hits == 0:
        suggested_check = "Review additional days/ports to confirm whether this is normal traffic behavior."

    answer = (
        f"Retrieved {len(retrieved.chunks)} evidence chunks (strategy {retrieved.strategy}) in {retrieved.latency_ms:.1f} ms. "
        f"{anomaly_hits} chunks contain anomaly flags."
    )

    return {
        "answer": answer,
        "evidence": evidence,
        "interpretation": interpretation,
        "suggested_check": suggested_check,
    }


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query thesis retrieval collection")
    parser.add_argument("--query", required=True)
    parser.add_argument("--strategy", default="A", choices=["A", "B", "C", "a", "b", "c"])
    parser.add_argument("--persist_dir", default="data/thesis_chroma")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--port", default=None)
    parser.add_argument("--date_from", default=None)
    parser.add_argument("--date_to", default=None)
    parser.add_argument("--vessel_type", default=None)
    parser.add_argument("--anomaly", default=None, help="true/false")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    retriever = ThesisRetriever(
        persist_dir=args.persist_dir,
        strategy=args.strategy,
        embedding_model=args.embedding_model,
    )
    filters = QueryFilters(
        port=args.port,
        date_from=args.date_from,
        date_to=args.date_to,
        vessel_type=args.vessel_type,
        anomaly=_parse_bool(args.anomaly),
    )
    out = retriever.retrieve(query=args.query, top_k=args.top_k, filters=filters)
    response = format_incident_aware_answer(query=args.query, retrieved=out)

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
