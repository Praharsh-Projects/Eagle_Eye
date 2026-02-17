"""Embed chunk strategies with sentence-transformers and build persistent Chroma collections."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.runtime import import_chromadb


STRATEGY_FILES = {
    "A": "strategy_a_event_chunks.jsonl",
    "B": "strategy_b_port_day_chunks.jsonl",
    "C": "strategy_c_hybrid_chunks.jsonl",
}


def _iter_jsonl_batches(path: Path, batch_size: int) -> Iterator[Tuple[List[str], List[str], List[Dict[str, Any]]]]:
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(str(row["id"]))
            texts.append(str(row["text"]))
            metadatas.append(_safe_metadata(dict(row.get("metadata", {}))))

            if len(ids) >= batch_size:
                yield ids, texts, metadatas
                ids, texts, metadatas = [], [], []

    if ids:
        yield ids, texts, metadatas


def _safe_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for fp in path.rglob("*"):
        if fp.is_file():
            total += fp.stat().st_size
    return round(total / (1024 * 1024), 3)


def _embed_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors


def build_index(
    chunks_dir: Path,
    persist_dir: Path,
    strategy: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    rebuild: bool = False,
) -> Dict[str, Any]:
    strategy = strategy.upper()
    if strategy not in STRATEGY_FILES:
        raise RuntimeError(f"Unknown strategy: {strategy}")

    chunk_file = chunks_dir / STRATEGY_FILES[strategy]
    if not chunk_file.exists():
        raise RuntimeError(f"Chunk file missing: {chunk_file}. Run src.thesis.chunking first.")

    model = SentenceTransformer(embedding_model)

    chromadb = import_chromadb()
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    collection_name = f"thesis_{strategy.lower()}"
    if rebuild:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    chunk_count = 0
    embedding_dim = 0
    embed_seconds = 0.0
    upsert_seconds = 0.0

    for id_batch, text_batch, meta_batch in _iter_jsonl_batches(chunk_file, batch_size=batch_size):
        start_embed = time.perf_counter()
        vectors = _embed_texts(model=model, texts=text_batch, batch_size=batch_size)
        embed_seconds += time.perf_counter() - start_embed
        if embedding_dim == 0 and vectors.size > 0:
            embedding_dim = int(vectors.shape[1])

        start_upsert = time.perf_counter()
        collection.upsert(
            ids=list(id_batch),
            documents=list(text_batch),
            metadatas=list(meta_batch),
            embeddings=vectors.tolist(),
        )
        upsert_seconds += time.perf_counter() - start_upsert
        chunk_count += len(id_batch)

    if chunk_count == 0:
        raise RuntimeError(f"No chunks found in {chunk_file}")

    manifest = {
        "strategy": strategy,
        "collection": collection_name,
        "embedding_model": embedding_model,
        "chunks": int(chunk_count),
        "embedding_dim": int(embedding_dim),
        "embed_seconds": round(embed_seconds, 4),
        "upsert_seconds": round(upsert_seconds, 4),
        "total_seconds": round(embed_seconds + upsert_seconds, 4),
        "persist_dir": str(persist_dir),
        "persist_size_mb": _dir_size_mb(persist_dir),
        "batch_size": int(batch_size),
    }

    with (persist_dir / f"manifest_{strategy.lower()}.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def build_all(
    chunks_dir: Path,
    persist_dir: Path,
    embedding_model: str,
    batch_size: int,
    rebuild: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for strategy in ["A", "B", "C"]:
        chunk_file = chunks_dir / STRATEGY_FILES[strategy]
        if not chunk_file.exists():
            continue
        out[strategy] = build_index(
            chunks_dir=chunks_dir,
            persist_dir=persist_dir,
            strategy=strategy,
            embedding_model=embedding_model,
            batch_size=batch_size,
            rebuild=rebuild,
        )
    return out


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed and index thesis chunks by strategy")
    parser.add_argument("--chunks_dir", default="data/thesis_chunks")
    parser.add_argument("--persist_dir", default="data/thesis_chroma")
    parser.add_argument("--strategy", default="all", choices=["A", "B", "C", "all", "a", "b", "c"])
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    strategy = args.strategy.upper()
    if strategy == "ALL":
        summary = build_all(
            chunks_dir=Path(args.chunks_dir),
            persist_dir=Path(args.persist_dir),
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )
    else:
        summary = build_index(
            chunks_dir=Path(args.chunks_dir),
            persist_dir=Path(args.persist_dir),
            strategy=strategy,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )
    print("Indexing completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
