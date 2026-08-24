"""Process-wide serialized Parquet reads for Streamlit's concurrent script threads."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pandas as pd


_PARQUET_READ_LOCK = threading.RLock()


def read_parquet_safely(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read one Parquet file without overlapping native Arrow readers."""
    with _PARQUET_READ_LOCK:
        return pd.read_parquet(path, **kwargs)
