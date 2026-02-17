"""Shared helpers for thesis pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_DEST_ALIASES: Dict[str, str] = {
    "RU LED": "RULED",
    "RULED": "RULED",
    "LT KLJ": "LTKLJ",
    "LTKLJ": "LTKLJ",
    "SE GOT": "SEGOT",
    "SEGOT": "SEGOT",
    "SE KAR": "SEKAR",
    "SEKAR": "SEKAR",
    "FI HEL": "FIHEL",
    "FIHEL": "FIHEL",
}


def normalize_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if re.match(r"^\d+\.0+$", text):
        return text.split(".")[0]
    return text


def normalize_destination(value: Any, aliases: Optional[Dict[str, str]] = None) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).upper().strip()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"

    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "UNKNOWN"

    alias_map = dict(DEFAULT_DEST_ALIASES)
    if aliases:
        alias_map.update(aliases)

    if text in alias_map:
        return alias_map[text]
    nospace = text.replace(" ", "")
    if nospace in alias_map:
        return alias_map[nospace]

    if re.match(r"^[A-Z]{2}\s[A-Z]{3}$", text):
        return text.replace(" ", "")
    if re.match(r"^[A-Z]{2}[A-Z]{3}$", nospace):
        return nospace
    return text


def normalize_locode(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper().replace(" ", "")
    if text in {"", "NAN", "NONE"}:
        return ""
    return text


def normalize_vessel_type(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return "unknown"
    return text


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_csv_with_schema(csv_path: Path, limit_rows: Optional[int] = None) -> pd.DataFrame:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().replace('"', "")
    cols = [c.strip() for c in header.split(",")]
    dtypes = {c: "string" for c in cols}
    return pd.read_csv(
        csv_path,
        dtype=dtypes,
        low_memory=False,
        nrows=limit_rows if limit_rows and limit_rows > 0 else None,
    )


def haversine_km(lat1: Sequence[float], lon1: Sequence[float], lat2: Sequence[float], lon2: Sequence[float]) -> np.ndarray:
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    rad = np.pi / 180.0
    lat1r = lat1 * rad
    lon1r = lon1 * rad
    lat2r = lat2 * rad
    lon2r = lon2 * rad

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0088 * c


def iter_batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]
