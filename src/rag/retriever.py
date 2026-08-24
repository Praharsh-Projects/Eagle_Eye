"""Retrieval layer for AIS traffic/docs collections with metadata + bbox filters."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from openai import OpenAI

from src.utils.config import load_config
from src.utils.serialization import normalize_destination, normalize_identifier
from src.utils.time import in_date_range
from src.utils.runtime import create_chroma_client, import_chromadb


def _normalize_vessel_type(value: str) -> str:
    return value.strip().lower()


def _normalize_nav_status(value: str) -> str:
    return value.strip().lower()


def _normalize_flag(value: str) -> str:
    return value.strip().upper()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None


def _first_present(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "nat"}:
            return text
    return None


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        mag_a += va * va
        mag_b += vb * vb
    if mag_a <= 0 or mag_b <= 0:
        return 1.0
    similarity = dot / (math.sqrt(mag_a) * math.sqrt(mag_b))
    return 1.0 - similarity


@dataclass
class QueryFilters:
    mmsi: Optional[str] = None
    imo: Optional[str] = None
    locode: Optional[str] = None
    port_name: Optional[str] = None
    vessel_type: Optional[str] = None
    flag: Optional[str] = None
    destination: Optional[str] = None
    nav_status: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None

    def normalized(self) -> "QueryFilters":
        return QueryFilters(
            mmsi=normalize_identifier(self.mmsi.strip()) if self.mmsi else None,
            imo=normalize_identifier(self.imo.strip()) if self.imo else None,
            locode=self.locode.strip().upper().replace(" ", "") if self.locode else None,
            port_name=self.port_name.strip().lower() if self.port_name else None,
            vessel_type=_normalize_vessel_type(self.vessel_type)
            if self.vessel_type
            else None,
            flag=_normalize_flag(self.flag) if self.flag else None,
            destination=normalize_destination(self.destination)
            if self.destination
            else None,
            nav_status=_normalize_nav_status(self.nav_status)
            if self.nav_status
            else None,
            date_from=self.date_from,
            date_to=self.date_to,
            lat_min=_safe_float(self.lat_min),
            lat_max=_safe_float(self.lat_max),
            lon_min=_safe_float(self.lon_min),
            lon_max=_safe_float(self.lon_max),
        )


@dataclass
class EvidenceItem:
    id: str
    text: str
    metadata: Dict[str, Any]
    source_kind: str
    distance: Optional[float] = None


@dataclass
class RetrievalResult:
    mode: str
    evidence: List[EvidenceItem]
    where_filter: Optional[Dict[str, Any]]
    backend: str = "unknown"

    @property
    def min_distance(self) -> Optional[float]:
        values = [item.distance for item in self.evidence if item.distance is not None]
        return min(values) if values else None


class RAGRetriever:
    def __init__(
        self,
        persist_dir: str | Path,
        config_path: str | Path = "config/config.yaml",
        top_k: Optional[int] = None,
    ) -> None:
        self.config = load_config(config_path)
        chromadb = import_chromadb()
        self.persist_dir = Path(persist_dir)
        self.client, self.vector_backend = create_chroma_client(
            chromadb=chromadb,
            persist_dir=self.persist_dir,
            config=self.config,
        )
        requested_provider = str(
            os.getenv(
                "EAGLE_EYE_RETRIEVAL_PROVIDER",
                self.config.get("retrieval", {}).get("provider", "auto"),
            )
        ).strip().lower()
        if requested_provider not in {"auto", "openai", "lexical"}:
            raise ValueError("EAGLE_EYE_RETRIEVAL_PROVIDER must be auto, openai, or lexical")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if requested_provider == "openai" and not api_key:
            raise RuntimeError("OpenAI retrieval was explicitly selected, but OPENAI_API_KEY is not set.")
        self.openai: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key and requested_provider != "lexical" else None
        self.query_backend = "openai_embedding" if self.openai is not None else "local_lexical"
        self.embedding_model = self.config["models"]["embedding_model"]
        self.top_k = int(top_k if top_k is not None else self.config["retrieval"].get("top_k", 5))
        self.prefilter_candidate_limit = int(
            self.config["retrieval"].get("bbox_candidate_limit", 20000)
        )
        self.local_scan_limit = int(self.config["retrieval"].get("local_lexical_scan_limit", 10000))
        # Query runtime is read-only: an absent collection is an initialization
        # error, never an excuse to create an empty index during a user query.
        self.traffic_collection = self.client.get_collection(
            name=self.config["index"]["traffic_collection"]
        )
        self.docs_collection = self.client.get_collection(
            name=self.config["index"]["docs_collection"]
        )
        metadata_override = os.getenv("TRAFFIC_METADATA_INDEX_PATH", "").strip()
        self.metadata_index_path = (
            Path(metadata_override)
            if metadata_override
            else self.persist_dir / "traffic_metadata_index.csv"
        )
        self._metadata_df: Optional[pd.DataFrame] = None
        self._docs_lexical_cache: Optional[
            tuple[List[str], List[str], List[Dict[str, Any]]]
        ] = None
        # The local traffic index is intentionally loaded once at startup.  A
        # query must not walk thousands of Chroma rows just to perform lexical
        # matching; filtering this slim dataframe is both faster and exact for
        # late coverage dates that do not occur in the first N vector rows.
        if self.openai is None and bool(
            self.config.get("retrieval", {}).get("preload_local_metadata", True)
        ):
            self._load_metadata_df()

    def _embed_query(self, question: str) -> List[float]:
        if self.openai is None:
            raise RuntimeError("Remote embeddings are unavailable in local lexical retrieval mode.")
        response = self.openai.embeddings.create(model=self.embedding_model, input=[question])
        return response.data[0].embedding

    @property
    def retrieval_backend(self) -> str:
        return f"{self.vector_backend}:{self.query_backend}"

    @staticmethod
    def _requested_top_k(top_k: Optional[int], default: int) -> int:
        return max(0, int(default if top_k is None else top_k))

    @staticmethod
    def _lexical_tokens(value: Any) -> List[str]:
        stopwords = {
            "a", "an", "and", "are", "at", "be", "between", "by", "for", "from", "how",
            "in", "is", "it", "of", "on", "or", "show", "the", "to", "was", "what", "which",
            "with", "were", "will",
        }
        tokens = re.findall(r"[a-z0-9][a-z0-9_.:-]*", str(value or "").lower())
        return [token for token in tokens if token not in stopwords and len(token) > 1]

    def _lexical_rank(
        self,
        *,
        question: str,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        source_kind: str,
        top_k: int,
        allow_structured_zero_score: bool = False,
    ) -> List[EvidenceItem]:
        if top_k <= 0:
            return []
        query_tokens = self._lexical_tokens(question)
        query_set = set(query_tokens)
        anchors = self._lexical_anchors(question)
        candidates: List[
            tuple[str, str, Dict[str, Any], Dict[str, int], str]
        ] = []
        document_frequency = {token: 0 for token in query_set}
        for index, doc_id in enumerate(ids):
            document = str(documents[index] if index < len(documents) else "")
            metadata = (
                metadatas[index]
                if index < len(metadatas) and isinstance(metadatas[index], dict)
                else {}
            )
            metadata_text = " ".join(
                str(metadata.get(field, ""))
                for field in (
                    "source_file",
                    "title",
                    "locode",
                    "locode_norm",
                    "port_name",
                    "port_name_norm",
                    "mmsi",
                    "imo",
                    "vessel_type",
                    "vessel_type_norm",
                    "destination",
                    "destination_norm",
                    "timestamp_date",
                    "timestamp_full",
                    "event_kind",
                )
            )
            candidate_text = f"{document} {metadata_text}".lower()
            # Explicit codes and identifiers are grounding constraints, not
            # optional ranking hints.  A SOLAS question must never be answered
            # from a NIS2 chunk merely because both contain "require".
            if anchors and any(
                re.search(rf"(?<![a-z0-9]){re.escape(anchor)}(?![a-z0-9])", candidate_text)
                is None
                for anchor in anchors
            ):
                continue
            candidate_tokens = self._lexical_tokens(candidate_text)
            token_counts: Dict[str, int] = {}
            for token in candidate_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            for token in query_set:
                if token in token_counts:
                    document_frequency[token] += 1
            candidates.append((str(doc_id), document, dict(metadata), token_counts, " ".join(candidate_tokens)))

        population = max(1, len(candidates))
        ranked: List[tuple[float, EvidenceItem]] = []
        for doc_id, document, metadata, token_counts, normalized_document in candidates:
            matched = [token for token in query_set if token in token_counts]
            score = sum(
                (math.log((population + 1.0) / (document_frequency[token] + 1.0)) + 1.0)
                * (1.0 + math.log1p(token_counts[token]))
                for token in matched
            )
            normalized_question = " ".join(query_tokens)
            if normalized_question and normalized_question in normalized_document:
                score += 4.0
            # Without a structured traffic scope or an explicit anchor, one
            # generic word is not enough to claim relevant source grounding.
            strong_single_match = any(
                len(token) >= 8
                and document_frequency[token] <= max(2, int(population * 0.1))
                for token in matched
            )
            if (
                score <= 0
                or (
                    not allow_structured_zero_score
                    and not anchors
                    and len(matched) < 2
                    and not strong_single_match
                )
            ):
                continue
            distance = 1.0 / (1.0 + max(0.0, score))
            ranked.append(
                (
                    score,
                    EvidenceItem(
                        id=str(doc_id),
                        text=document,
                        metadata=dict(metadata),
                        source_kind=source_kind,
                        distance=distance,
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1].id))
        return [item for _, item in ranked[:top_k]]

    @staticmethod
    def _lexical_anchors(question: str) -> List[str]:
        """Return explicit identifiers that every accepted chunk must contain."""

        text = str(question or "")
        known = {
            "emsa",
            "eur-lex",
            "ilo",
            "imo",
            "isps",
            "locode",
            "mmsi",
            "nis2",
            "solas",
        }
        anchors = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
            if token.lower() in known
        }
        for token in re.findall(r"\b[A-Z][A-Z0-9]{2,11}\b", text):
            lowered = token.lower()
            if any(char.isdigit() for char in token) or len(token) == 5:
                if lowered not in {"about", "march", "show", "today", "which"}:
                    anchors.add(lowered)
        anchors.update(match.lower() for match in re.findall(r"\b(?:19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?\b", text))
        anchors.update(re.findall(r"\b\d{6,9}\b", text))
        return sorted(anchors)

    def _get_collection_candidates(
        self,
        collection: Any,
        *,
        where: Optional[Dict[str, Any]],
        max_rows: int,
    ) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while len(ids) < max_rows:
            limit = min(page_size, max_rows - len(ids))
            payload = collection.get(
                where=where,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"],
            )
            page_ids = list(payload.get("ids") or [])
            if not page_ids:
                break
            page_documents = list(payload.get("documents") or [])
            page_metadatas = list(payload.get("metadatas") or [])
            ids.extend(str(value) for value in page_ids)
            documents.extend(str(value or "") for value in page_documents)
            metadatas.extend(dict(value or {}) for value in page_metadatas)
            if len(page_ids) < limit:
                break
            offset += len(page_ids)
        return ids, documents, metadatas

    def _load_metadata_df(self) -> Optional[pd.DataFrame]:
        if self._metadata_df is not None:
            return self._metadata_df
        if not self.metadata_index_path.exists():
            return None
        wanted = {
            "stable_id",
            "mmsi",
            "imo",
            "flag_norm",
            "vessel_type_norm",
            "nav_status_norm",
            "destination_norm",
            "port_name_norm",
            "locode_norm",
            "timestamp_date",
            "timestamp_full",
            "latitude",
            "longitude",
            "source_file",
            "event_kind",
        }
        # The production metadata CSV is hundreds of MB. Loading both the full
        # serialized event and its bounded excerpt inflated the UI process to
        # several GB. Prefer the excerpt and use full text only for a legacy
        # index that does not contain excerpts.
        available = set(pd.read_csv(self.metadata_index_path, nrows=0).columns)
        text_field = (
            "serialized_excerpt"
            if "serialized_excerpt" in available
            else "serialized_text"
            if "serialized_text" in available
            else None
        )
        selected = wanted & available
        if text_field:
            selected.add(text_field)
        string_columns = {
            # Object strings avoid invoking PyArrow's native scalar comparison
            # path from Streamlit's script thread (a reproducible macOS
            # segfault), while the bounded column set keeps memory controlled.
            name: object
            for name in selected
            if name not in {"latitude", "longitude"}
        }
        self._metadata_df = pd.read_csv(
            self.metadata_index_path,
            usecols=lambda name: name in selected,
            dtype=string_columns,
            low_memory=True,
        )
        return self._metadata_df

    @staticmethod
    def _sample_frame(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
        if limit <= 0 or len(frame) <= limit:
            return frame
        stride = max(1, math.ceil(len(frame) / limit))
        return frame.iloc[::stride].head(limit)

    @staticmethod
    def _python_text_values(series: pd.Series) -> List[str]:
        """Convert scalars without constructing pandas' Arrow string array."""

        return ["" if pd.isna(value) else str(value) for value in series.tolist()]

    @staticmethod
    def _traffic_document_from_metadata(metadata: Dict[str, Any]) -> str:
        """Create a faithful excerpt when an older metadata index has no text column."""

        labels = (
            ("timestamp", "timestamp_full"),
            ("date", "timestamp_date"),
            ("MMSI", "mmsi"),
            ("IMO", "imo"),
            ("LOCODE", "locode_norm"),
            ("port", "port_name_norm"),
            ("vessel type", "vessel_type_norm"),
            ("destination", "destination_norm"),
            ("navigation status", "nav_status_norm"),
            ("latitude", "latitude"),
            ("longitude", "longitude"),
            ("event", "event_kind"),
            ("source", "source_file"),
        )
        values: List[str] = []
        for label, field in labels:
            value = metadata.get(field)
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "<na>"}:
                values.append(f"{label}: {text}")
        return "AIS event | " + " | ".join(values)

    def _prefilter_candidate_frame(self, filters: QueryFilters) -> Optional[pd.DataFrame]:
        df = self._load_metadata_df()
        if df is None or df.empty:
            return None
        mask = self._metadata_filter_mask(df, filters)
        return df.loc[mask]

    @staticmethod
    def _metadata_filter_mask(df: pd.DataFrame, filters: QueryFilters) -> pd.Series:
        f = filters.normalized()
        mask = pd.Series(True, index=df.index)
        if f.mmsi and "mmsi" in df.columns:
            mask &= df["mmsi"].eq(f.mmsi)
        if f.imo and "imo" in df.columns:
            mask &= df["imo"].eq(f.imo)
        if f.locode and "locode_norm" in df.columns:
            mask &= df["locode_norm"].eq(f.locode)
        if f.port_name and "port_name_norm" in df.columns:
            mask &= df["port_name_norm"].eq(f.port_name)
        if f.vessel_type and "vessel_type_norm" in df.columns:
            mask &= df["vessel_type_norm"].eq(f.vessel_type)
        if f.flag and "flag_norm" in df.columns:
            mask &= df["flag_norm"].eq(f.flag)
        if f.destination and "destination_norm" in df.columns:
            mask &= df["destination_norm"].eq(f.destination)
        if f.nav_status and "nav_status_norm" in df.columns:
            mask &= df["nav_status_norm"].eq(f.nav_status)
        if f.date_from and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].ge(f.date_from)
        if f.date_to and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].le(f.date_to)
        if f.lat_min is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") >= f.lat_min
        if f.lat_max is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") <= f.lat_max
        if f.lon_min is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") >= f.lon_min
        if f.lon_max is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") <= f.lon_max
        return mask.fillna(False)

    def _metadata_df_from_collection(
        self,
        filters: Optional[QueryFilters] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        where = self._build_where(filters or QueryFilters())
        rows: List[Dict[str, Any]] = []
        offset = 0
        batch_size = 2000
        target_rows = int(max_rows or self.prefilter_candidate_limit)

        while len(rows) < target_rows:
            limit = min(batch_size, target_rows - len(rows))
            got = self.traffic_collection.get(
                where=where,
                limit=limit,
                offset=offset,
                include=["metadatas"],
            )
            metas = got.get("metadatas", []) or []
            if not metas:
                break
            for metadata in metas:
                if isinstance(metadata, dict):
                    rows.append(metadata)
            if len(metas) < limit:
                break
            offset += len(metas)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _has_bbox_filter(self, filters: QueryFilters) -> bool:
        f = filters.normalized()
        return any(
            value is not None
            for value in (f.lat_min, f.lat_max, f.lon_min, f.lon_max)
        )

    def _build_where(self, filters: QueryFilters) -> Optional[Dict[str, Any]]:
        f = filters.normalized()
        clauses: List[Dict[str, Any]] = []
        if f.mmsi:
            clauses.append({"mmsi": {"$eq": f.mmsi}})
        if f.imo:
            clauses.append({"imo": {"$eq": f.imo}})
        if f.locode:
            clauses.append({"locode_norm": {"$eq": f.locode}})
        if f.port_name:
            clauses.append({"port_name_norm": {"$eq": f.port_name}})
        if f.vessel_type:
            clauses.append({"vessel_type_norm": {"$eq": f.vessel_type}})
        if f.flag:
            clauses.append({"flag_norm": {"$eq": f.flag}})
        if f.destination:
            clauses.append({"destination_norm": {"$eq": f.destination}})
        if f.nav_status:
            clauses.append({"nav_status_norm": {"$eq": f.nav_status}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _matches_filters(self, metadata: Dict[str, Any], filters: QueryFilters) -> bool:
        f = filters.normalized()
        if f.mmsi and str(metadata.get("mmsi", "")).strip() != f.mmsi:
            return False
        if f.imo and str(metadata.get("imo", "")).strip() != f.imo:
            return False
        if f.locode and str(metadata.get("locode_norm", "")).upper() != f.locode:
            return False
        if f.port_name and str(metadata.get("port_name_norm", "")).lower() != f.port_name:
            return False
        if f.vessel_type and str(metadata.get("vessel_type_norm", "")).lower() != f.vessel_type:
            return False
        if f.flag and str(metadata.get("flag_norm", "")).upper() != f.flag:
            return False
        if f.destination and str(metadata.get("destination_norm", "")).upper() != f.destination:
            return False
        if f.nav_status and str(metadata.get("nav_status_norm", "")).lower() != f.nav_status:
            return False
        if not in_date_range(
            str(metadata.get("timestamp_date", "")),
            date_from=f.date_from,
            date_to=f.date_to,
        ):
            return False

        lat = _safe_float(metadata.get("latitude"))
        lon = _safe_float(metadata.get("longitude"))
        if f.lat_min is not None and (lat is None or lat < f.lat_min):
            return False
        if f.lat_max is not None and (lat is None or lat > f.lat_max):
            return False
        if f.lon_min is not None and (lon is None or lon < f.lon_min):
            return False
        if f.lon_max is not None and (lon is None or lon > f.lon_max):
            return False
        return True

    def _prefilter_candidate_ids(
        self,
        filters: QueryFilters,
        *,
        limit: Optional[int] = None,
    ) -> Optional[List[str]]:
        df = self._load_metadata_df()
        if df is None or df.empty:
            return None
        filtered = df[self._metadata_filter_mask(df, filters)]
        if filtered.empty or "stable_id" not in filtered.columns:
            return []
        ids = filtered["stable_id"].astype(str).tolist()
        effective_limit = self.prefilter_candidate_limit if limit is None else max(0, int(limit))
        return ids[:effective_limit] if effective_limit else ids

    def _rank_candidates_by_similarity(
        self, query_embedding: Sequence[float], candidate_ids: Sequence[str], top_k: int
    ) -> List[EvidenceItem]:
        ranked: List[EvidenceItem] = []
        for id_batch in _batched(list(candidate_ids), batch_size=512):
            got = self.traffic_collection.get(
                ids=list(id_batch),
                include=["documents", "metadatas", "embeddings"],
            )
            ids = got.get("ids", [])
            docs = got.get("documents", [])
            metas = got.get("metadatas", [])
            embeddings = got.get("embeddings", [])
            for idx, doc_id in enumerate(ids):
                emb = embeddings[idx] if idx < len(embeddings) else None
                if emb is None:
                    continue
                distance = _cosine_distance(query_embedding, emb)
                ranked.append(
                    EvidenceItem(
                        id=doc_id,
                        text=docs[idx] if idx < len(docs) else "",
                        metadata=metas[idx] if idx < len(metas) else {},
                        source_kind="traffic",
                        distance=distance,
                    )
                )
        ranked.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
        return ranked[:top_k]

    def query_traffic(
        self, question: str, filters: QueryFilters, top_k: Optional[int] = None
    ) -> RetrievalResult:
        where = self._build_where(filters)
        requested_top_k = self._requested_top_k(top_k, self.top_k)
        if requested_top_k == 0:
            return RetrievalResult(
                mode="traffic",
                evidence=[],
                where_filter=where,
                backend=self.retrieval_backend,
            )
        # Local lexical traffic queries use the metadata index directly.  Do
        # not pay Chroma's cold `count()` cost on the analytics request path.
        available = (
            len(self._metadata_df)
            if self.openai is None and self._metadata_df is not None
            else self.traffic_collection.count()
        )
        if available == 0:
            return RetrievalResult(
                mode="traffic",
                evidence=[],
                where_filter=where,
                backend=self.retrieval_backend,
            )

        if self.openai is None:
            candidate_frame = self._prefilter_candidate_frame(filters)
            if candidate_frame is not None:
                if candidate_frame.empty:
                    return RetrievalResult(
                        mode="traffic",
                        evidence=[],
                        where_filter={"prefilter_candidates": 0},
                        backend=self.retrieval_backend,
                    )
                total_candidates = len(candidate_frame)
                candidate_frame = self._sample_frame(candidate_frame, self.local_scan_limit)
                ids = self._python_text_values(candidate_frame["stable_id"])
                metadata_fields = [
                    field
                    for field in candidate_frame.columns
                    if field not in {"stable_id", "serialized_excerpt", "serialized_text"}
                ]
                metadatas = candidate_frame[metadata_fields].to_dict(orient="records")
                text_field = next(
                    (
                        field
                        for field in ("serialized_excerpt", "serialized_text")
                        if field in candidate_frame.columns
                    ),
                    None,
                )
                if text_field:
                    documents = self._python_text_values(candidate_frame[text_field])
                else:
                    documents = [
                        self._traffic_document_from_metadata(metadata)
                        for metadata in metadatas
                    ]
            else:
                total_candidates = min(available, self.local_scan_limit)
                ids, documents, metadatas = self._get_collection_candidates(
                    self.traffic_collection,
                    where=where,
                    max_rows=total_candidates,
                )
                kept = [
                    index
                    for index, metadata in enumerate(metadatas)
                    if self._matches_filters(metadata, filters)
                ]
                ids = [ids[index] for index in kept]
                documents = [documents[index] for index in kept]
                metadatas = [metadatas[index] for index in kept]
            evidence = self._lexical_rank(
                question=question,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                source_kind="traffic",
                top_k=requested_top_k,
                allow_structured_zero_score=where is not None,
            )
            return RetrievalResult(
                mode="traffic",
                evidence=evidence,
                where_filter={"prefilter_candidates": total_candidates},
                backend=self.retrieval_backend,
            )

        query_embedding = self._embed_query(question)

        # Bounding-box filtering is enforced through pandas prefilter before ranking.
        if self._has_bbox_filter(filters):
            candidate_ids = self._prefilter_candidate_ids(filters)
            if candidate_ids is None:
                # Metadata index unavailable; fallback to vector query + post-filtering.
                n_results = min(requested_top_k * 5, available)
                try:
                    result = self.traffic_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=where,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    result = self.traffic_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=None,
                        include=["documents", "metadatas", "distances"],
                    )
                hits = self._to_evidence(result, source_kind="traffic")
                filtered = [item for item in hits if self._matches_filters(item.metadata, filters)]
                filtered.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
                return RetrievalResult(
                    mode="traffic",
                    evidence=filtered[:requested_top_k],
                    where_filter={"prefilter_candidates": "metadata-index-missing"},
                    backend=self.retrieval_backend,
                )
            if not candidate_ids:
                return RetrievalResult(
                    mode="traffic",
                    evidence=[],
                    where_filter={"prefilter_candidates": 0},
                    backend=self.retrieval_backend,
                )
            evidence = self._rank_candidates_by_similarity(
                query_embedding, candidate_ids, requested_top_k
            )
            evidence = [item for item in evidence if self._matches_filters(item.metadata, filters)]
            return RetrievalResult(
                mode="traffic",
                evidence=evidence,
                where_filter={"prefilter_candidates": len(candidate_ids)},
                backend=self.retrieval_backend,
            )

        n_results = min(requested_top_k * 3, available)
        try:
            result = self.traffic_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            result = self.traffic_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=None,
                include=["documents", "metadatas", "distances"],
            )
        hits = self._to_evidence(result, source_kind="traffic")
        filtered = [item for item in hits if self._matches_filters(item.metadata, filters)]
        filtered.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
        return RetrievalResult(
            mode="traffic",
            evidence=filtered[:requested_top_k],
            where_filter=where,
            backend=self.retrieval_backend,
        )

    def query_docs(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        requested_top_k = self._requested_top_k(top_k, self.top_k)
        if requested_top_k == 0:
            return RetrievalResult(
                mode="docs",
                evidence=[],
                where_filter=None,
                backend=self.retrieval_backend,
            )
        if self.openai is None:
            if self._docs_lexical_cache is None:
                self._docs_lexical_cache = self._get_collection_candidates(
                    self.docs_collection,
                    where=None,
                    max_rows=self.local_scan_limit,
                )
            ids, documents, metadatas = self._docs_lexical_cache
            if not ids:
                return RetrievalResult(
                    mode="docs",
                    evidence=[],
                    where_filter=None,
                    backend=self.retrieval_backend,
                )
            evidence = self._lexical_rank(
                question=question,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                source_kind="docs",
                top_k=requested_top_k,
            )
            return RetrievalResult(
                mode="docs",
                evidence=evidence,
                where_filter=None,
                backend=self.retrieval_backend,
            )

        available = self.docs_collection.count()
        if available == 0:
            return RetrievalResult(
                mode="docs",
                evidence=[],
                where_filter=None,
                backend=self.retrieval_backend,
            )
        query_embedding = self._embed_query(question)
        n_results = min(requested_top_k, available)
        result = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits = self._to_evidence(result, source_kind="docs")
        hits.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
        return RetrievalResult(
            mode="docs",
            evidence=hits[:requested_top_k],
            where_filter=None,
            backend=self.retrieval_backend,
        )

    def _to_evidence(self, result: Dict[str, Any], source_kind: str) -> List[EvidenceItem]:
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        evidence: List[EvidenceItem] = []
        for idx, doc_id in enumerate(ids):
            evidence.append(
                EvidenceItem(
                    id=doc_id,
                    text=docs[idx] if idx < len(docs) else "",
                    metadata=metas[idx] if idx < len(metas) else {},
                    source_kind=source_kind,
                    distance=distances[idx] if idx < len(distances) else None,
                )
            )
        return evidence

    @staticmethod
    def is_aggregation_question(question: str) -> bool:
        q = question.lower()
        keywords = (
            "how many",
            "count",
            "number of",
            "total",
            "aggregate",
            "sum of",
        )
        return any(token in q for token in keywords)

    @staticmethod
    def is_jump_detection_question(question: str) -> bool:
        q = question.lower()
        keywords = ("sudden", "jump", "teleport", "coordinate jump", "suspicious ais")
        return any(token in q for token in keywords)

    def compute_traffic_count(
        self, filters: QueryFilters, question: str, max_ids: int = 200
    ) -> Optional[Dict[str, Any]]:
        if not self.is_aggregation_question(question):
            return None
        df = self._load_metadata_df()
        if df is None or df.empty:
            df = self._metadata_df_from_collection(filters=filters, max_rows=max_ids * 100)
        if df.empty:
            return {"analysis_type": "count", "count": 0, "rows": []}

        f = filters.normalized()
        mask = pd.Series(True, index=df.index)
        if f.mmsi and "mmsi" in df.columns:
            mask &= df["mmsi"].astype(str).str.strip() == f.mmsi
        if f.imo and "imo" in df.columns:
            mask &= df["imo"].astype(str).str.strip() == f.imo
        if f.locode and "locode_norm" in df.columns:
            mask &= (
                df["locode_norm"].astype(str).str.upper().str.replace(" ", "", regex=False)
                == f.locode
            )
        if f.port_name and "port_name_norm" in df.columns:
            mask &= df["port_name_norm"].astype(str).str.lower() == f.port_name
        if f.vessel_type and "vessel_type_norm" in df.columns:
            mask &= df["vessel_type_norm"].astype(str).str.lower() == f.vessel_type
        if f.flag and "flag_norm" in df.columns:
            mask &= df["flag_norm"].astype(str).str.upper() == f.flag
        if f.destination and "destination_norm" in df.columns:
            mask &= df["destination_norm"].astype(str).str.upper() == f.destination
        if f.nav_status and "nav_status_norm" in df.columns:
            mask &= df["nav_status_norm"].astype(str).str.lower() == f.nav_status
        if f.date_from and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].astype(str) >= f.date_from
        if f.date_to and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].astype(str) <= f.date_to
        if f.lat_min is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") >= f.lat_min
        if f.lat_max is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") <= f.lat_max
        if f.lon_min is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") >= f.lon_min
        if f.lon_max is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") <= f.lon_max

        filtered = df[mask]
        ids = filtered["stable_id"].astype(str).tolist()[:max_ids]
        return {"analysis_type": "count", "count": int(len(filtered)), "rows": ids}

    def detect_sudden_jumps(
        self,
        filters: QueryFilters,
        max_minutes: int = 30,
        km_threshold: float = 20.0,
        speed_kn_threshold: float = 40.0,
        min_distance_km_for_speed_rule: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Detect likely suspicious jumps for a single MMSI within time window.
        """
        df = self._load_metadata_df()
        if df is None or df.empty:
            df = self._metadata_df_from_collection(filters=filters, max_rows=50000)
        if df.empty:
            return {
                "analysis_type": "jump_detection",
                "count": 0,
                "rows": [],
                "events": [],
                "scope_status": "unsupported" if filters.locode else "not_requested",
                "scope_applied": False,
                "reason": (
                    "No AIS position rows with an observed port/LOCODE field were available; destination values were not used as a location proxy."
                    if filters.locode
                    else "No AIS position rows were available."
                ),
            }

        f = filters.normalized()
        work = df.copy()
        if "event_kind" in work.columns:
            work = work[work["event_kind"].fillna("").astype(str) == "ais_position"]

        scope_metadata: Dict[str, Any] = {
            "scope_status": "not_requested",
            "scope_applied": False,
            "scope_field": None,
            "requested_locode": f.locode,
        }
        if f.locode:
            scope_field = None
            for field in ("locode_norm", "locode", "port_key"):
                if field not in work.columns:
                    continue
                normalized = (
                    work[field]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.replace(r"[^A-Z0-9]", "", regex=True)
                    .replace({"NA": "", "NAN": "", "NONE": "", "NULL": "", "UNK": "", "UNKNOWN": ""})
                )
                if not normalized.ne("").any():
                    continue
                work = work[normalized == f.locode].copy()
                scope_field = field
                break
            if scope_field is None:
                return {
                    "analysis_type": "jump_detection",
                    "count": 0,
                    "rows": [],
                    "events": [],
                    "scope_status": "unsupported",
                    "scope_applied": False,
                    "requested_locode": f.locode,
                    "reason": (
                        f"AIS position rows do not contain a populated observed port/LOCODE field for {f.locode}. "
                        "Destination values were not used as a location proxy."
                    ),
                }
            scope_metadata.update(
                {
                    "scope_status": "applied",
                    "scope_applied": True,
                    "scope_field": scope_field,
                }
            )
        if "mmsi" in work.columns:
            work["mmsi"] = work["mmsi"].astype(str).map(normalize_identifier)
        if f.mmsi and "mmsi" in work.columns:
            work = work[work["mmsi"] == f.mmsi]
        if "timestamp_date" in work.columns:
            if f.date_from:
                work = work[work["timestamp_date"].astype(str) >= f.date_from]
            if f.date_to:
                work = work[work["timestamp_date"].astype(str) <= f.date_to]
        work["latitude"] = pd.to_numeric(work.get("latitude"), errors="coerce")
        work["longitude"] = pd.to_numeric(work.get("longitude"), errors="coerce")
        work["timestamp_dt"] = pd.to_datetime(
            work.get("timestamp_full", work.get("date", pd.Series(dtype=str))),
            errors="coerce",
            utc=True,
        )
        if "timestamp_date" not in work.columns:
            work["timestamp_date"] = work["timestamp_dt"].dt.strftime("%Y-%m-%d")
            if f.date_from:
                work = work[work["timestamp_date"].astype(str) >= f.date_from]
            if f.date_to:
                work = work[work["timestamp_date"].astype(str) <= f.date_to]
        if "mmsi" not in work.columns:
            work["mmsi"] = None
        work = work.dropna(subset=["timestamp_dt", "latitude", "longitude", "mmsi"])
        if work.empty:
            return {
                "analysis_type": "jump_detection",
                "count": 0,
                "rows": [],
                "events": [],
                **scope_metadata,
            }

        work = work.sort_values(["mmsi", "timestamp_dt"])
        jump_ids: List[str] = []
        jump_events: List[Dict[str, Any]] = []
        for _, group in work.groupby("mmsi"):
            g = group.copy()
            g["prev_timestamp_dt"] = g["timestamp_dt"].shift(1)
            g["prev_latitude"] = g["latitude"].shift(1)
            g["prev_longitude"] = g["longitude"].shift(1)
            dt_minutes = (g["timestamp_dt"] - g["prev_timestamp_dt"]).dt.total_seconds() / 60.0
            dlat = g["latitude"] - g["prev_latitude"]
            dlon = g["longitude"] - g["prev_longitude"]
            # Approx rough km distance on Earth surface.
            dist_km = ((dlat * 111.0) ** 2 + (dlon * 111.0) ** 2) ** 0.5
            implied_speed_kn = ((dist_km / (dt_minutes / 60.0)) / 1.852).replace(
                [float("inf"), float("-inf")], pd.NA
            )
            mask = (
                (dt_minutes > 0)
                & (dt_minutes <= max_minutes)
                & (
                    (dist_km >= km_threshold)
                    | (
                        (dist_km >= min_distance_km_for_speed_rule)
                        & (implied_speed_kn >= speed_kn_threshold)
                    )
                )
            )
            ids = g.loc[mask, "stable_id"].astype(str).tolist()
            jump_ids.extend(ids)
            if mask.any():
                flagged = g.loc[mask].copy()
                flagged["dt_minutes"] = dt_minutes.loc[mask].astype(float)
                flagged["distance_km"] = dist_km.loc[mask].astype(float)
                flagged["implied_speed_kn"] = implied_speed_kn.loc[mask].astype(float)
                flagged["trigger_rule"] = flagged.apply(
                    lambda row: (
                        "distance_threshold"
                        if float(row.get("distance_km", 0.0)) >= km_threshold
                        else "speed_threshold"
                    ),
                    axis=1,
                )
                for _, row in flagged.head(200).iterrows():
                    jump_events.append(
                        {
                            "stable_id": str(row.get("stable_id", "")),
                            "mmsi": str(row.get("mmsi", "")),
                            "timestamp_full": str(row.get("timestamp_full", "")),
                            "latitude": _safe_float(row.get("latitude")),
                            "longitude": _safe_float(row.get("longitude")),
                            "prev_latitude": _safe_float(row.get("prev_latitude")),
                            "prev_longitude": _safe_float(row.get("prev_longitude")),
                            "dt_minutes": float(row.get("dt_minutes", 0.0)),
                            "distance_km": float(row.get("distance_km", 0.0)),
                            "implied_speed_kn": float(row.get("implied_speed_kn", 0.0)),
                            "trigger_rule": str(row.get("trigger_rule", "")),
                            "port": _first_present(
                                row.get("locode_norm"),
                                row.get("locode"),
                                row.get("destination_norm"),
                                row.get("port_name_norm"),
                            ),
                        }
                    )
        return {
            "analysis_type": "jump_detection",
            "count": len(jump_ids),
            "rows": jump_ids[:200],
            "events": jump_events[:200],
            **scope_metadata,
        }
