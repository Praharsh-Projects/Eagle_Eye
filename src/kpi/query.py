"""Deterministic analytics engine over KPI materialized tables."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.parquet_io import read_parquet_safely


PORT_ALIAS_TO_CODE: Dict[str, str] = {
    "gothenburg": "SEGOT",
    "goteborg": "SEGOT",
    "goteborgs": "SEGOT",
    "port of gothenburg": "SEGOT",
    "helsinki": "FIHEL",
    "port of helsinki": "FIHEL",
    "sodertalje": "SESOE",
    "port of sodertalje": "SESOE",
    "karlshamn": "SEKAN",
    "port of karlshamn": "SEKAN",
    "karlskrona": "SEKAA",
    "port of karlskrona": "SEKAA",
    "gdansk": "PLGDN",
    "port of gdansk": "PLGDN",
    "gdynia": "PLGDY",
    "klaipeda": "LTKLJ",
    "riga": "LVRIX",
    "kotka": "FIKTK",
    "swinoujscie": "PLSWI",
    "szczecin": "PLSZZ",
    "ventspils": "LVVNT",
}

MEASUREMENT_CONTRACT_V2: Dict[str, str] = {
    "version": "measurement_contract.v2",
    "historical_arrival_count": "valid_port_call_events",
    "daily_distinct_vessel_presence": "distinct_mmsi_per_port_day",
    "completed_dwell": "0 < departure_minus_arrival <= 45_days",
}


def _normalize_vessel_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().lower()


def _normalize_port_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip()


def _normalize_text_token(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^port of\s+", "", text)
    return text


def _normalize_mmsi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None


def _normalize_source_scope(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"port_call", "port_calls"}:
        return "port_call"
    if token in {"ais", "ais_destination_proxy", "ais_proxy", "ais_derived"}:
        return "ais_destination_proxy"
    return None


def _as_date_str(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _parse_window(window: Optional[str], max_date: Optional[pd.Timestamp]) -> tuple[Optional[str], Optional[str]]:
    if not window or max_date is None or pd.isna(max_date):
        return None, None
    m = re.match(r"last_(\d{1,2})_weeks", window)
    if not m:
        return None, None
    weeks = int(m.group(1))
    end = pd.Timestamp(max_date).floor("D")
    start = end - pd.Timedelta(days=7 * weeks)
    return _as_date_str(start), _as_date_str(end)


def _with_arrival_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose measurement_contract.v2 fields without removing legacy columns.

    Historical arrival analytics count valid port-call events.  The older
    ``arrivals_vessels`` column is a per-bucket distinct-vessel feature and is
    retained for pressure and forecast compatibility, but it is never used as
    the event total.
    """

    if frame.empty:
        return frame
    work = frame.copy()
    if "arrivals_vessels" in work.columns and "daily_distinct_vessels" not in work.columns:
        work["daily_distinct_vessels"] = pd.to_numeric(
            work["arrivals_vessels"], errors="coerce"
        )
    authority = "arrivals_events" if "arrivals_events" in work.columns else "arrivals_vessels"
    if authority in work.columns:
        work["arrival_count"] = pd.to_numeric(work[authority], errors="coerce")
    return work


@dataclass
class AnalyticsResult:
    status: str
    answer: str
    table: Optional[pd.DataFrame]
    chart: Optional[pd.DataFrame]
    coverage_notes: List[str]
    caveats: List[str]


class KPIQueryEngine:
    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)
        self._arrivals_daily: Optional[pd.DataFrame] = None
        self._arrivals_hourly: Optional[pd.DataFrame] = None
        self._dwell: Optional[pd.DataFrame] = None
        self._occupancy: Optional[pd.DataFrame] = None
        self._congestion: Optional[pd.DataFrame] = None
        self._port_catalog: Optional[pd.DataFrame] = None
        self._voyages: Optional[pd.DataFrame] = None
        self._caps: Optional[Dict[str, Any]] = None

    def _load_parquet(self, name: str) -> pd.DataFrame:
        path = self.processed_dir / name
        if not path.exists():
            return pd.DataFrame()
        return read_parquet_safely(path)

    def preload(self) -> "KPIQueryEngine":
        """Materialize UI-critical tables before sharing this engine across threads."""
        _ = self.arrivals_daily
        _ = self.arrivals_hourly
        _ = self.dwell
        _ = self.occupancy
        _ = self.congestion
        _ = self.port_catalog
        _ = self.voyages
        _ = self.capabilities()
        return self

    @property
    def arrivals_daily(self) -> pd.DataFrame:
        if self._arrivals_daily is None:
            df = self._load_parquet("arrivals_daily.parquet")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
            self._arrivals_daily = _with_arrival_semantics(df)
        if self._arrivals_daily is not None and "arrival_count" not in self._arrivals_daily.columns:
            self._arrivals_daily = _with_arrival_semantics(self._arrivals_daily)
        return self._arrivals_daily

    @property
    def arrivals_hourly(self) -> pd.DataFrame:
        if self._arrivals_hourly is None:
            df = self._load_parquet("arrivals_hourly.parquet")
            if "datetime_hour" in df.columns:
                df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce", utc=True).dt.floor("h")
            self._arrivals_hourly = _with_arrival_semantics(df)
        if self._arrivals_hourly is not None and "arrival_count" not in self._arrivals_hourly.columns:
            self._arrivals_hourly = _with_arrival_semantics(self._arrivals_hourly)
        return self._arrivals_hourly

    @property
    def dwell(self) -> pd.DataFrame:
        if self._dwell is None:
            df = self._load_parquet("dwell_time.parquet")
            for col in ("arrival_time", "departure_time", "arrival_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            if "dwell_minutes" in df.columns:
                duration = pd.to_numeric(df["dwell_minutes"], errors="coerce")
                df = df[(duration > 0) & (duration <= 45 * 24 * 60)].copy()
                df["dwell_minutes"] = duration.loc[df.index]
            self._dwell = df
        return self._dwell

    @property
    def occupancy(self) -> pd.DataFrame:
        if self._occupancy is None:
            df = self._load_parquet("occupancy_hourly.parquet")
            if "datetime_hour" in df.columns:
                df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce", utc=True).dt.floor("h")
            self._occupancy = df
        return self._occupancy

    @property
    def congestion(self) -> pd.DataFrame:
        if self._congestion is None:
            df = self._load_parquet("congestion_daily.parquet")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
            self._congestion = df
        return self._congestion

    @property
    def port_catalog(self) -> pd.DataFrame:
        if self._port_catalog is None:
            df = self._load_parquet("port_catalog.parquet")
            for col in ("first_seen", "last_seen"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.floor("D")
            self._port_catalog = df
        return self._port_catalog

    def capabilities(self) -> Dict[str, Any]:
        if self._caps is not None:
            return self._caps
        caps_path = self.processed_dir / "kpi_capabilities.json"
        if caps_path.exists():
            with caps_path.open("r", encoding="utf-8") as f:
                self._caps = json.load(f)
        else:
            self._caps = {
                "has_port_calls": bool((self.arrivals_daily.get("source_kind") == "port_call").any())
                if not self.arrivals_daily.empty
                else False,
                "has_ais_destination_proxy": bool((self.arrivals_daily.get("source_kind") == "ais_destination_proxy").any())
                if not self.arrivals_daily.empty
                else False,
                "has_dwell_time": not self.dwell.empty,
                "has_occupancy_hourly": not self.occupancy.empty,
            }
        self._caps.setdefault("measurement_contract", dict(MEASUREMENT_CONTRACT_V2))
        return self._caps

    @property
    def voyages(self) -> pd.DataFrame:
        if self._voyages is None:
            df = self._load_parquet("voyages.parquet")
            for col in ("departure_time", "arrival_time"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            self._voyages = df
        return self._voyages

    def coverage_notes(self, df: pd.DataFrame, date_col: str) -> List[str]:
        notes: List[str] = []
        if df.empty:
            notes.append("No rows available for the requested filters.")
            return notes
        start = pd.to_datetime(df[date_col], errors="coerce", utc=True).min()
        end = pd.to_datetime(df[date_col], errors="coerce", utc=True).max()
        if pd.notna(start) and pd.notna(end):
            notes.append(f"Coverage window: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        if "source_kind" in df.columns:
            sources = sorted(set(df["source_kind"].fillna("unknown").astype(str)))
            notes.append("Data sources used: " + ", ".join(sources))
        notes.append(f"Rows used: {len(df):,}")
        freshness = self.data_freshness_date()
        if freshness:
            notes.append(f"Historical data freshness: latest available observation is {freshness}")
        return notes

    def data_freshness_date(self) -> Optional[str]:
        caps = self.capabilities()
        value = caps.get("date_max") if isinstance(caps, dict) else None
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(parsed):
            return pd.Timestamp(parsed).strftime("%Y-%m-%d")
        if not self.arrivals_daily.empty and "date" in self.arrivals_daily.columns:
            latest = pd.to_datetime(self.arrivals_daily["date"], errors="coerce", utc=True).max()
            if pd.notna(latest):
                return pd.Timestamp(latest).strftime("%Y-%m-%d")
        return None

    def has_current_data(self, reference_date: Optional[str] = None) -> bool:
        freshness = pd.to_datetime(self.data_freshness_date(), errors="coerce", utc=True)
        target = pd.to_datetime(reference_date, errors="coerce", utc=True) if reference_date else pd.Timestamp.now(tz="UTC")
        if pd.isna(freshness) or pd.isna(target):
            return False
        return pd.Timestamp(freshness).floor("D") >= pd.Timestamp(target).floor("D")

    def resolve_port_token(self, port_token: Optional[str]) -> Optional[str]:
        token = _normalize_port_token(port_token)
        if not token:
            return None
        if re.fullmatch(r"[A-Za-z]{2}\s?[A-Za-z]{3}", token):
            return token.upper().replace(" ", "")

        norm = _normalize_text_token(token)
        alias_code = PORT_ALIAS_TO_CODE.get(norm)
        if alias_code:
            return alias_code

        catalog = self.port_catalog
        if catalog.empty:
            return token

        code = token.upper().replace(" ", "")
        work = catalog.copy()
        for col in ("port_key", "locode_norm", "port_label", "port_name_norm"):
            if col not in work.columns:
                work[col] = ""
            work[col] = work[col].fillna("").astype(str)
        if "arrivals_total" not in work.columns:
            work["arrivals_total"] = 0
        work["arrivals_total"] = pd.to_numeric(work["arrivals_total"], errors="coerce").fillna(0)
        work["source_kind"] = work.get("source_kind", "").fillna("").astype(str).str.lower()
        work["locode_norm"] = work.get("locode_norm", "").fillna("").astype(str).str.upper()
        work["is_structured_port"] = (
            (work["source_kind"] == "port_call")
            & work["locode_norm"].str.fullmatch(r"[A-Z]{5}")
        )

        exact_code = work[
            (work["port_key"].str.upper() == code) | (work["locode_norm"].str.upper() == code)
        ]
        if not exact_code.empty:
            row = exact_code.sort_values("arrivals_total", ascending=False).iloc[0]
            return str(row.get("port_key") or row.get("locode_norm") or token).strip()

        work["port_label_norm"] = work["port_label"].map(_normalize_text_token)
        work["port_name_norm_clean"] = work["port_name_norm"].map(_normalize_text_token)
        contains = work[
            work["port_label_norm"].str.contains(norm, regex=False)
            | work["port_name_norm_clean"].str.contains(norm, regex=False)
        ]
        if not contains.empty:
            if contains["is_structured_port"].any():
                contains = contains[contains["is_structured_port"]]
            row = contains.sort_values("arrivals_total", ascending=False).iloc[0]
            return str(row.get("port_key") or row.get("locode_norm") or token).strip()

        def _best_similarity(row: pd.Series) -> float:
            cand_a = str(row.get("port_label_norm", ""))
            cand_b = str(row.get("port_name_norm_clean", ""))
            return max(
                SequenceMatcher(None, norm, cand_a).ratio() if cand_a else 0.0,
                SequenceMatcher(None, norm, cand_b).ratio() if cand_b else 0.0,
            )

        work["similarity"] = work.apply(_best_similarity, axis=1)
        fuzzy = work[work["similarity"] >= 0.80]
        if not fuzzy.empty:
            if fuzzy["is_structured_port"].any():
                fuzzy = fuzzy[fuzzy["is_structured_port"]]
            row = fuzzy.sort_values(["similarity", "arrivals_total"], ascending=[False, False]).iloc[0]
            return str(row.get("port_key") or row.get("locode_norm") or token).strip()

        return token

    def is_known_port_token(self, port_token: Optional[str]) -> bool:
        token = _normalize_port_token(port_token)
        if not token:
            return False
        norm = _normalize_text_token(token)
        if norm in PORT_ALIAS_TO_CODE:
            return True

        catalog = self.port_catalog
        if catalog.empty:
            return False
        code = token.upper().replace(" ", "")
        work = catalog.copy()
        for col in ("port_key", "locode_norm", "port_label", "port_name_norm"):
            if col not in work.columns:
                work[col] = ""
            work[col] = work[col].fillna("").astype(str)
        key_match = (
            (work["port_key"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True) == code)
            | (work["locode_norm"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True) == code)
        )
        label_match = (
            work["port_label"].map(_normalize_text_token).eq(norm)
            | work["port_name_norm"].map(_normalize_text_token).eq(norm)
        )
        return bool((key_match | label_match).any())

    def _filter_port(self, df: pd.DataFrame, port: Optional[str]) -> pd.DataFrame:
        token = _normalize_port_token(port)
        if not token or df.empty:
            return df
        resolved = self.resolve_port_token(token) or token
        code = resolved.upper().replace(" ", "")
        raw_norm = _normalize_text_token(token)
        resolved_norm = _normalize_text_token(resolved)

        mask = pd.Series(False, index=df.index)
        if "port_key" in df.columns:
            mask |= df["port_key"].fillna("").astype(str).str.upper() == code
        if "locode_norm" in df.columns:
            mask |= df["locode_norm"].fillna("").astype(str).str.upper() == code
        if "port_name_norm" in df.columns:
            port_names = df["port_name_norm"].fillna("").astype(str).map(_normalize_text_token)
            mask |= port_names == raw_norm
            mask |= port_names == resolved_norm
        if "port_label" in df.columns:
            label_base = (
                df["port_label"]
                .fillna("")
                .astype(str)
                .str.replace(r"\s*\([^)]*\)\s*", "", regex=True)
                .map(_normalize_text_token)
            )
            mask |= label_base == raw_norm
            mask |= label_base == resolved_norm

        filtered = df[mask]
        return filtered

    def _filter_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        date_from: Optional[str],
        date_to: Optional[str],
        window: Optional[str] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        date_series = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        out = df.copy()

        if (not date_from or not date_to) and window:
            win_from, win_to = _parse_window(window, date_series.max())
            date_from = date_from or win_from
            date_to = date_to or win_to

        if date_from:
            out = out[date_series >= pd.Timestamp(date_from, tz="UTC")]
            date_series = pd.to_datetime(out[date_col], errors="coerce", utc=True)
        if date_to:
            end_ts = pd.to_datetime(date_to, errors="coerce", utc=True)
            if pd.notna(end_ts):
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date_to).strip()):
                    out = out[date_series < pd.Timestamp(end_ts).floor("D") + pd.Timedelta(days=1)]
                else:
                    out = out[date_series <= pd.Timestamp(end_ts)]
        return out

    def _filter_vessel_type(self, df: pd.DataFrame, vessel_type: Optional[str]) -> pd.DataFrame:
        vt = _normalize_vessel_type(vessel_type)
        if not vt or df.empty or "vessel_type_norm" not in df.columns:
            return df
        return df[df["vessel_type_norm"].fillna("").astype(str).str.contains(vt, regex=False)]

    def _filter_source_scope(self, df: pd.DataFrame, source_scope: Optional[str]) -> pd.DataFrame:
        scope = _normalize_source_scope(source_scope)
        if not scope or df.empty or "source_kind" not in df.columns:
            return df
        return df[df["source_kind"].fillna("").astype(str).str.lower() == scope]

    def _prefer_arrival_source(
        self,
        df: pd.DataFrame,
        time_col: str,
        allow_day_gap_fallback: bool = False,
    ) -> pd.DataFrame:
        if df.empty or "source_kind" not in df.columns:
            return df

        work = df.copy()
        work["source_kind"] = work["source_kind"].fillna("").astype(str).str.lower()
        if "port_key" not in work.columns or not (work["source_kind"] == "port_call").any():
            return work

        # Structured port-call records are the numeric authority. For an
        # unfiltered aggregate at one resolved port, AIS destination rows may
        # fill a whole day absent from the port-call source. Category-filtered
        # counts never use that fallback because proxy vessel labels are not an
        # equivalent event taxonomy.
        if allow_day_gap_fallback and time_col in work.columns:
            structured = work[work["source_kind"] == "port_call"]
            fallback = work[work["source_kind"] != "port_call"]
            structured_days = set(
                pd.to_datetime(structured[time_col], errors="coerce", utc=True)
                .dt.floor("D")
                .dropna()
                .tolist()
            )
            fallback_days = pd.to_datetime(fallback[time_col], errors="coerce", utc=True).dt.floor("D")
            fallback = fallback[~fallback_days.isin(structured_days)]
            return pd.concat([structured, fallback], ignore_index=False).sort_index()

        structured_ports = set(
            work.loc[work["source_kind"] == "port_call", "port_key"].dropna().astype(str)
        )
        structured = work[
            (work["source_kind"] == "port_call")
            & work["port_key"].fillna("").astype(str).isin(structured_ports)
        ]
        fallback = work[
            (work["source_kind"] != "port_call")
            & ~work["port_key"].fillna("").astype(str).isin(structured_ports)
        ]
        return pd.concat([structured, fallback], ignore_index=False).sort_index()

    @staticmethod
    def _arrivals_source_note(df: pd.DataFrame, source_scope: Optional[str]) -> str:
        scope = _normalize_source_scope(source_scope)
        if scope == "port_call":
            return "Arrivals are filtered to `port_call` rows from the structured port-call source only."
        if scope == "ais_destination_proxy":
            return "Arrivals are filtered to `ais_destination_proxy` rows only."
        if df.empty or "source_kind" not in df.columns:
            return "Arrivals are based on the matched deterministic KPI rows."

        sources = {str(item).strip().lower() for item in df["source_kind"].dropna().astype(str).tolist() if str(item).strip()}
        if sources == {"port_call"}:
            return "Arrivals are computed from structured port-call rows for the matched scope."
        if sources == {"ais_destination_proxy"}:
            return "Arrivals are computed from AIS destination proxy rows because structured port-call rows were not available in the matched scope."
        return "Arrivals use port-call rows where available and AIS destination proxy only for unmatched periods in the same scope."

    @staticmethod
    def _filter_voyage_endpoint(df: pd.DataFrame, endpoint: str, port_token: Optional[str]) -> pd.DataFrame:
        token = _normalize_port_token(port_token)
        if not token or df.empty:
            return df
        key_col = f"{endpoint}_port_key"
        label_col = f"{endpoint}_port_label"
        code = token.upper().replace(" ", "")
        low = token.lower()
        mask = pd.Series(False, index=df.index)
        if key_col in df.columns:
            mask |= df[key_col].fillna("").astype(str).str.upper() == code
        if label_col in df.columns:
            mask |= df[label_col].fillna("").astype(str).str.lower().str.contains(low, regex=False)
        return df[mask]

    @staticmethod
    def _filter_dow(df: pd.DataFrame, date_col: str, dow: Optional[str]) -> pd.DataFrame:
        if not dow or df.empty:
            return df
        norm = dow.strip().title()
        work = df.copy()
        dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
        if norm == "Weekend":
            work = work[dates.dt.day_name().isin(["Saturday", "Sunday"])]
        elif norm == "Weekday":
            work = work[~dates.dt.day_name().isin(["Saturday", "Sunday"])]
        else:
            work = work[dates.dt.day_name() == norm]
        return work

    @staticmethod
    def unsupported(reason: str) -> AnalyticsResult:
        return AnalyticsResult(
            status="unsupported",
            answer="I don't have evidence in the dataset to answer that.",
            table=None,
            chart=None,
            coverage_notes=[],
            caveats=[reason],
        )

    @staticmethod
    def no_data(hint: str) -> AnalyticsResult:
        return AnalyticsResult(
            status="no_data",
            answer="I don't have evidence in the dataset to answer that.",
            table=None,
            chart=None,
            coverage_notes=[],
            caveats=[hint],
        )

    def no_current_data(self, requested_scope: str = "the requested current period") -> AnalyticsResult:
        freshness = self.data_freshness_date() or "unknown"
        return AnalyticsResult(
            status="no_current_data",
            answer=(
                f"Current operational data is not available for {requested_scope}. "
                f"The latest historical observation is {freshness}."
            ),
            table=None,
            chart=None,
            coverage_notes=[f"Historical data freshness: latest available observation is {freshness}"],
            caveats=["Historical Eagle Eye data must not be presented as current operational truth."],
        )

    def get_arrivals(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        window: Optional[str] = None,
        source_scope: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing. Run `python -m src.kpi.build_kpis ...` first.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end, window=window)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_dow(work, "date", dow)
        work = self._filter_source_scope(work, source_scope)
        work = self._prefer_arrival_source(
            work,
            "date",
            allow_day_gap_fallback=False,
        )

        if work.empty:
            return self.no_data("No arrival rows matched these filters. Broaden port/date/vessel-type constraints.")

        daily = (
            work.groupby("date", dropna=False)
            .agg(
                arrival_count=("arrival_count", "sum"),
                daily_distinct_vessels=("daily_distinct_vessels", "sum"),
                arrivals_vessels=("arrivals_vessels", "sum"),
                arrivals_events=("arrivals_events", "sum"),
            )
            .reset_index()
            .sort_values("date")
        )
        total_arrivals = int(daily["arrival_count"].sum())
        answer = (
            f"Matched {total_arrivals:,} vessel arrivals across {len(daily):,} day buckets"
            + (f" for {port}" if port else "")
            + "."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=daily,
            chart=daily.set_index("date")[["arrival_count"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[self._arrivals_source_note(work, source_scope)],
        )

    def get_arrival_composition(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        source_scope: Optional[str] = None,
    ) -> AnalyticsResult:
        """Return vessel-type shares for a filtered arrival scope."""
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")
        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        work = self._filter_source_scope(work, source_scope)
        work = self._prefer_arrival_source(work, "date")
        if work.empty:
            return self.no_data("No arrival rows matched the requested vessel-type composition scope.")

        composition = (
            work.assign(
                vessel_type_norm=work["vessel_type_norm"].fillna("unknown").astype(str).replace("", "unknown")
            )
            .groupby("vessel_type_norm", dropna=False)
            .agg(arrival_count=("arrival_count", "sum"))
            .reset_index()
        )
        total = float(composition["arrival_count"].sum())
        if total <= 0:
            return self.no_data("Matched arrival rows did not contain a positive vessel count.")
        composition["share_percent"] = composition["arrival_count"] / total * 100.0
        composition["category"] = composition["vessel_type_norm"]
        composition["value"] = composition["arrival_count"].astype(float)
        composition = composition.sort_values(
            ["arrival_count", "vessel_type_norm"], ascending=[False, True]
        ).reset_index(drop=True)
        leader = composition.iloc[0]
        answer = (
            f"{leader['vessel_type_norm']} is the largest vessel-type arrival group"
            + (f" at {port}" if port else "")
            + f", with {int(leader['arrival_count']):,} arrivals ({float(leader['share_percent']):.1f}% of {int(total):,})."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=composition,
            chart=composition.set_index("vessel_type_norm")[["arrival_count"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[self._arrivals_source_note(work, source_scope)],
        )

    def get_peak_arrival_day(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        window: Optional[str] = None,
        source_scope: Optional[str] = None,
    ) -> AnalyticsResult:
        base = self.get_arrivals(
            port=port,
            start=start,
            end=end,
            vessel_type=vessel_type,
            dow=None,
            window=window,
            source_scope=source_scope,
        )
        if base.status != "ok" or base.table is None or base.table.empty:
            return base

        daily = base.table.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce", utc=True).dt.floor("D")
        daily = daily.dropna(subset=["date"]).sort_values("date")
        if daily.empty:
            return self.no_data("No daily rows were available to compute a peak day.")

        peak = daily.loc[daily["arrival_count"].idxmax()]
        peak_date = peak["date"].strftime("%Y-%m-%d")
        peak_val = int(peak["arrival_count"])

        answer = (
            f"Peak arrivals day is {peak_date} with {peak_val:,} vessel arrivals"
            + (f" for {port}" if port else "")
            + "."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=daily,
            chart=daily.set_index("date")[["arrival_count"]],
            coverage_notes=base.coverage_notes + [f"Peak day computed from {len(daily):,} day buckets."],
            caveats=base.caveats,
        )

    def top_ports_by_arrivals(
        self,
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        top_n: int = 10,
        source_scope: Optional[str] = None,
        country_codes: Optional[Sequence[str]] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")
        work = self._filter_dates(df, "date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_dow(work, "date", dow)
        work = self._filter_source_scope(work, source_scope)
        work = self._prefer_arrival_source(work, "date")
        normalized_countries = {
            str(code or "").strip().upper()
            for code in (country_codes or [])
            if len(str(code or "").strip()) == 2
        }
        if normalized_countries:
            port_codes = work.get("locode_norm", work.get("port_key", pd.Series("", index=work.index)))
            work = work[
                port_codes.fillna("").astype(str).str.upper().str[:2].isin(normalized_countries)
            ]
        if work.empty:
            return self.no_data("No arrivals available for this time range/filter.")

        top = (
            work.groupby(["port_key", "port_label"], dropna=False)
            .agg(arrival_count=("arrival_count", "sum"))
            .reset_index()
            .sort_values(["arrival_count", "port_key"], ascending=[False, True])
            .head(max(1, top_n))
        )
        answer = f"Top {len(top)} ports by arrivals were computed for the selected filters."
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=top,
            chart=top.set_index("port_label")[["arrival_count"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[self._arrivals_source_note(work, source_scope)],
        )

    def get_busiest_dow(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        source_scope: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")
        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_source_scope(work, source_scope)
        work = self._prefer_arrival_source(work, "date")
        if work.empty:
            return self.no_data("No rows available for day-of-week analysis.")

        dates = pd.to_datetime(work["date"], errors="coerce", utc=True)
        by_day = (
            work.assign(day_of_week=dates.dt.day_name())
            .groupby("day_of_week", dropna=False)
            .agg(arrival_count=("arrival_count", "sum"))
            .reset_index()
            .sort_values("arrival_count", ascending=False)
        )
        busiest = by_day.iloc[0]
        answer = (
            f"Busiest weekday is {busiest['day_of_week']} with {int(busiest['arrival_count']):,} arrivals"
            + (f" for {port}" if port else "")
            + "."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_day,
            chart=by_day.set_index("day_of_week")[["arrival_count"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[self._arrivals_source_note(work, source_scope)],
        )

    def compare_weekdays(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        day_a: str,
        day_b: str,
        vessel_type: Optional[str] = None,
        source_scope: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_source_scope(work, source_scope)
        work = self._prefer_arrival_source(work, "date")
        if work.empty:
            return self.no_data("No rows available for weekday comparison.")

        dates = pd.to_datetime(work["date"], errors="coerce", utc=True)
        by_day = (
            work.assign(day_of_week=dates.dt.day_name())
            .groupby("day_of_week", dropna=False)
            .agg(arrival_count=("arrival_count", "sum"))
            .reset_index()
        )
        day_a_title = day_a.title()
        day_b_title = day_b.title()
        pair = by_day[by_day["day_of_week"].isin([day_a_title, day_b_title])].copy()
        if pair.empty or len(pair) < 2:
            return self.no_data(f"Could not find both weekdays ({day_a_title}, {day_b_title}) in the filtered window.")

        a_val = float(pair[pair["day_of_week"] == day_a_title]["arrival_count"].iloc[0])
        b_val = float(pair[pair["day_of_week"] == day_b_title]["arrival_count"].iloc[0])
        if a_val > b_val:
            winner = day_a_title
            ratio = (a_val / max(b_val, 1.0))
        else:
            winner = day_b_title
            ratio = (b_val / max(a_val, 1.0))

        answer = (
            f"{winner} is busier in the filtered history. "
            f"{day_a_title}={int(a_val):,} vs {day_b_title}={int(b_val):,} arrivals "
            f"(~{ratio:.2f}x)."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=pair.sort_values("arrival_count", ascending=False),
            chart=pair.set_index("day_of_week")[["arrival_count"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[self._arrivals_source_note(work, source_scope)],
        )

    def get_busiest_hour(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_hourly
        if df.empty:
            return self.no_data("arrivals_hourly.parquet is missing.")
        work = self._filter_port(df, port)
        work = self._filter_dates(work, "datetime_hour", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._prefer_arrival_source(work, "datetime_hour")
        if work.empty:
            return self.no_data("No rows available for hourly pattern analysis.")

        hours = pd.to_datetime(work["datetime_hour"], errors="coerce", utc=True)
        by_hour = (
            work.assign(hour=hours.dt.hour)
            .groupby("hour", dropna=False)
            .agg(arrival_count=("arrival_count", "sum"))
            .reset_index()
            .sort_values("arrival_count", ascending=False)
        )
        top = by_hour.iloc[0]
        answer = (
            f"Busiest hour is {int(top['hour']):02d}:00 UTC with {int(top['arrival_count']):,} arrivals"
            + (f" for {port}" if port else "")
            + "."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_hour,
            chart=by_hour.set_index("hour")[["arrival_count"]],
            coverage_notes=self.coverage_notes(work, "datetime_hour"),
            caveats=[self._arrivals_source_note(work, None)],
        )

    def get_arrival_weekday_hour_pattern(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
    ) -> AnalyticsResult:
        if self.arrivals_hourly.empty:
            return self.no_data("arrivals_hourly.parquet is missing.")
        work = self._filter_port(self.arrivals_hourly, port)
        work = self._filter_dates(work, "datetime_hour", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._prefer_arrival_source(work, "datetime_hour")
        if work.empty:
            return self.no_data("No rows are available for a weekday/hour arrival pattern.")

        timestamps = pd.to_datetime(work["datetime_hour"], errors="coerce", utc=True)
        pattern = (
            work.assign(day_of_week=timestamps.dt.day_name(), hour=timestamps.dt.hour)
            .dropna(subset=["day_of_week", "hour"])
            .groupby(["day_of_week", "hour"], dropna=False)
            .agg(arrival_count=("arrival_count", "sum"))
            .reset_index()
        )
        weekday_order = {
            day: index
            for index, day in enumerate(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
        }
        pattern["weekday_order"] = pattern["day_of_week"].map(weekday_order)
        pattern = pattern.sort_values(["weekday_order", "hour"], kind="stable").drop(columns="weekday_order")
        if pattern.empty:
            return self.no_data("No valid weekday/hour buckets are available after filtering.")
        peak = pattern.loc[pattern["arrival_count"].idxmax()]
        answer = (
            f"Peak historical arrival bucket for {port or 'the selected scope'} is "
            f"{peak['day_of_week']} at {int(peak['hour']):02d}:00 UTC with "
            f"{int(peak['arrival_count']):,} arrivals."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=pattern.reset_index(drop=True),
            chart=pattern.reset_index(drop=True),
            coverage_notes=self.coverage_notes(work, "datetime_hour"),
            caveats=[self._arrivals_source_note(work, None)],
        )

    def get_arrivals_dwell_correlation(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
    ) -> AnalyticsResult:
        if self.congestion.empty:
            return self.no_data("congestion_daily.parquet is missing.")
        work = self._filter_port(self.congestion, port)
        work = self._filter_dates(work, "date", start, end)
        work = self._prefer_arrival_source(work, "date")
        if "has_dwell" in work.columns:
            work = work[work["has_dwell"].fillna(False)]
        required = {"date", "arrivals_events", "median_dwell_minutes"}
        if work.empty or not required.issubset(work.columns):
            return self.no_data("Paired arrivals and dwell observations are unavailable for this scope.")

        daily = (
            work.assign(
                arrival_count=pd.to_numeric(work["arrivals_events"], errors="coerce"),
                median_dwell_minutes=pd.to_numeric(work["median_dwell_minutes"], errors="coerce"),
            )
            .dropna(subset=["arrival_count", "median_dwell_minutes"])
            .groupby("date", dropna=False)
            .agg(
                arrival_count=("arrival_count", "sum"),
                median_dwell_minutes=("median_dwell_minutes", "median"),
            )
            .reset_index()
            .sort_values("date")
        )
        if len(daily) < 3 or daily["arrival_count"].nunique() < 2 or daily["median_dwell_minutes"].nunique() < 2:
            return self.no_data("At least three varying paired daily observations are required for correlation.")
        correlation = float(daily["arrival_count"].corr(daily["median_dwell_minutes"]))
        answer = (
            f"The historical Pearson association between daily arrivals and median dwell at "
            f"{port or 'the selected scope'} is r={correlation:.3f} across {len(daily):,} paired days. "
            "This is an association, not evidence of causation."
        )
        summary = pd.DataFrame(
            [{"metric": "pearson_r", "value": correlation, "paired_days": int(len(daily))}]
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=summary,
            chart=daily.set_index("date")[["arrival_count", "median_dwell_minutes"]],
            coverage_notes=self.coverage_notes(daily, "date"),
            caveats=[
                "Correlation measures historical association only and does not establish causation.",
                "Dwell medians use only days with dwell observations.",
            ],
        )

    def get_avg_dwell_time(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        aggregation: Optional[str] = None,
    ) -> AnalyticsResult:
        if self.dwell.empty:
            return self.unsupported("Dwell-time analysis requires port-call arrival and departure timestamps.")

        work = self._filter_port(self.dwell, port)
        work = self._filter_dates(work, "arrival_date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_dow(work, "arrival_date", dow)
        if work.empty:
            return self.no_data("No dwell rows matched these filters.")

        median_dwell = float(work["dwell_minutes"].median())
        mean_dwell = float(work["dwell_minutes"].mean())
        port_clause = f" for {port}" if port else ""
        if aggregation == "mean":
            answer = f"Mean completed dwell time is {mean_dwell / 60.0:.6f} hours{port_clause}."
        elif aggregation == "median":
            answer = f"Median completed dwell time is {median_dwell / 60.0:.6f} hours{port_clause}."
        else:
            answer = (
                f"Median dwell time is {median_dwell:.1f} minutes; mean dwell is {mean_dwell:.1f} minutes"
                + port_clause
                + "."
            )

        by_type = (
            work.groupby("vessel_type_norm", dropna=False)
            .agg(
                calls=("mmsi", "size"),
                median_dwell_minutes=("dwell_minutes", "median"),
                mean_dwell_minutes=("dwell_minutes", "mean"),
            )
            .reset_index()
            .sort_values("calls", ascending=False)
        )
        by_type["complete_dwell_count"] = int(len(work))
        by_type["mean_dwell_hours"] = mean_dwell / 60.0
        by_type["median_dwell_hours"] = median_dwell / 60.0
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_type,
            chart=None,
            coverage_notes=self.coverage_notes(work, "arrival_date"),
            caveats=[],
        )

    def get_dwell_distribution(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
    ) -> AnalyticsResult:
        """Return an explicit histogram plus robust dwell summary statistics."""
        if self.dwell.empty:
            return self.unsupported("Dwell-time analysis requires port-call arrival and departure timestamps.")
        work = self._filter_port(self.dwell, port)
        work = self._filter_dates(work, "arrival_date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        values = pd.to_numeric(work.get("dwell_minutes"), errors="coerce").dropna()
        values = values[values >= 0]
        if values.empty:
            return self.no_data("No dwell values matched the requested distribution scope.")

        q25 = float(values.quantile(0.25))
        median = float(values.median())
        q75 = float(values.quantile(0.75))
        p90 = float(values.quantile(0.90))
        iqr = max(q75 - q25, 0.0)
        if iqr > 0:
            width = 2.0 * iqr / np.cbrt(len(values))
            bins = int(np.ceil((float(values.max()) - float(values.min())) / max(width, 1.0)))
        else:
            bins = 10
        bins = max(8, min(bins, 40))
        counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bins)
        histogram = pd.DataFrame(
            {
                "bin_start_minutes": edges[:-1],
                "bin_end_minutes": edges[1:],
                "bin_midpoint_minutes": (edges[:-1] + edges[1:]) / 2.0,
                "calls": counts.astype(int),
            }
        )
        answer = (
            f"Dwell distribution uses {len(values):,} port calls"
            + (f" at {port}" if port else "")
            + f": median {median:.1f} minutes, middle 50% {q25:.1f}-{q75:.1f} minutes, p90 {p90:.1f} minutes."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=histogram,
            chart=histogram.set_index("bin_midpoint_minutes")[["calls"]],
            coverage_notes=self.coverage_notes(work.loc[values.index], "arrival_date")
            + [f"Distribution sample: {len(values):,} calls", f"Histogram bins: {bins}"],
            caveats=["Histogram binning uses the Freedman-Diaconis rule, bounded to 8-40 bins."],
        )

    def get_mmsi_port_stays(
        self,
        mmsi: str,
        start: Optional[str],
        end: Optional[str],
        port: Optional[str] = None,
    ) -> AnalyticsResult:
        if self.dwell.empty:
            return self.no_data("dwell_time.parquet is missing.")

        norm_mmsi = _normalize_mmsi(mmsi)
        if not norm_mmsi:
            return self.no_data("Invalid MMSI provided.")

        work = self._filter_port(self.dwell, port)
        work = self._filter_dates(work, "arrival_date", start, end)
        if "mmsi" in work.columns:
            mmsi_series = work["mmsi"].astype(str).map(_normalize_mmsi)
            work = work[mmsi_series == norm_mmsi]

        if work.empty:
            return self.no_data("No port-stay rows matched this MMSI and date range.")

        cols = [
            c
            for c in [
                "mmsi",
                "port_key",
                "port_label",
                "arrival_time",
                "departure_time",
                "dwell_minutes",
                "vessel_type_norm",
                "source_kind",
            ]
            if c in work.columns
        ]
        table = (
            work[cols]
            .copy()
            .sort_values("arrival_time" if "arrival_time" in cols else cols[0])
            .reset_index(drop=True)
        )
        total_calls = len(table)
        total_hours = float(pd.to_numeric(table["dwell_minutes"], errors="coerce").fillna(0).sum() / 60.0)
        median_minutes = float(pd.to_numeric(table["dwell_minutes"], errors="coerce").median())

        if total_calls == 1:
            row = table.iloc[0]
            port_name = str(row.get("port_label") or row.get("port_key") or port or "the matched port")
            arrival = pd.to_datetime(row.get("arrival_time"), errors="coerce", utc=True)
            departure = pd.to_datetime(row.get("departure_time"), errors="coerce", utc=True)
            interval = ""
            if pd.notna(arrival) and pd.notna(departure):
                interval = (
                    f", from {arrival.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                    f"to {departure.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
            answer = (
                f"MMSI {norm_mmsi} was in port at {port_name} for {median_minutes:.1f} minutes "
                f"({total_hours:.1f} hours){interval}."
            )
        else:
            answer = (
                f"MMSI {norm_mmsi} had {total_calls:,} matched port call(s), "
                f"median dwell {median_minutes:.1f} minutes, total dwell {total_hours:.1f} hours."
            )
            if port:
                answer = answer[:-1] + f" for {port}."

        chart = None
        if {"arrival_time", "dwell_minutes"}.issubset(table.columns):
            chart = (
                table.assign(arrival_time=pd.to_datetime(table["arrival_time"], errors="coerce", utc=True))
                .dropna(subset=["arrival_time"])
                .sort_values("arrival_time")
                .set_index("arrival_time")[["dwell_minutes"]]
            )

        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=table,
            chart=chart,
            coverage_notes=self.coverage_notes(work, "arrival_date"),
            caveats=["Duration reflects arrival-to-departure time from port-call records."],
        )

    def get_congestion(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        dow: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.congestion
        if df.empty:
            return self.no_data("congestion_daily.parquet is missing.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end, window=window)
        work = self._filter_dow(work, "date", dow)
        if work.empty:
            return self.no_data("No congestion rows matched these filters.")

        caveats: List[str] = []
        if "pressure_kind" in work.columns:
            kinds = set(work["pressure_kind"].dropna().astype(str))
            if "full" in kinds and "partial_arrivals_proxy" in kinds:
                excluded = int((work["pressure_kind"] == "partial_arrivals_proxy").sum())
                work = work[work["pressure_kind"] == "full"].copy()
                caveats.append(
                    f"Excluded {excluded:,} arrivals-only pressure row(s) so full and partial pressure were not mixed."
                )
            pressure_kind = "full" if "full" in set(work["pressure_kind"].astype(str)) else "partial_arrivals_proxy"
        else:
            pressure_kind = "legacy"

        by_day = (
            work.groupby("date", dropna=False)
            .agg(
                congestion_index=("congestion_index", "mean"),
                arrivals_vessels=("arrivals_vessels", "sum"),
                median_dwell_minutes=("median_dwell_minutes", "median"),
                pressure_kind=("pressure_kind", "first") if "pressure_kind" in work.columns else ("source_kind", "first"),
            )
            .reset_index()
            .sort_values("date")
        )
        mean_ci = float(by_day["congestion_index"].mean())
        max_row = by_day.loc[by_day["congestion_index"].idxmax()]

        if len(by_day) == 1:
            level = "above" if mean_ci > 1.0 else ("below" if mean_ci < 1.0 else "at")
            port_label = f" at {port}" if port else ""
            label = "port pressure index"
            answer = (
                f"The {label}{port_label} on {max_row['date'].strftime('%Y-%m-%d')} is "
                f"{mean_ci:.2f}, which is {level} the 1.00 historical baseline."
            )
        else:
            label = "port pressure index"
            answer = (
                f"Average {label} is {mean_ci:.2f}; the highest-pressure day is "
                f"{max_row['date'].strftime('%Y-%m-%d')} at {max_row['congestion_index']:.2f}."
            )

        if pressure_kind == "partial_arrivals_proxy":
            caveats.append(
                "This is a pressure_v2 arrivals-only proxy; it must not be compared with the full arrivals-plus-dwell index."
            )

        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_day,
            chart=by_day.set_index("date")[["congestion_index"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=caveats,
        )

    def get_peak_congestion_days(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        *,
        dow: Optional[str] = None,
        window: Optional[str] = None,
        limit: int = 1,
    ) -> AnalyticsResult:
        result = self.get_congestion(port=port, start=start, end=end, dow=dow, window=window)
        if result.status != "ok" or result.table is None or result.table.empty:
            return result

        limit = max(1, min(int(limit), 20))
        ranked = result.table.sort_values("congestion_index", ascending=False).head(limit).reset_index(drop=True)
        if limit == 1:
            row = ranked.iloc[0]
            answer = (
                f"The highest-pressure day{f' at {port}' if port else ''} was "
                f"{pd.Timestamp(row['date']).strftime('%Y-%m-%d')} with a pressure index of "
                f"{float(row['congestion_index']):.2f}."
            )
        else:
            details = ", ".join(
                f"{pd.Timestamp(row['date']).strftime('%Y-%m-%d')} ({float(row['congestion_index']):.2f})"
                for _, row in ranked.iterrows()
            )
            answer = f"Top {len(ranked)} high-pressure days{f' at {port}' if port else ''}: {details}."
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=ranked,
            chart=ranked.set_index("date")[["congestion_index"]],
            coverage_notes=list(result.coverage_notes),
            caveats=list(result.caveats),
        )

    def compare_congestion_weekdays(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        day_a: str,
        day_b: str,
    ) -> AnalyticsResult:
        df = self.congestion
        if df.empty:
            return self.no_data("congestion_daily.parquet is missing.")
        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        if work.empty:
            return self.no_data("No pressure rows matched the requested weekday comparison.")

        caveats = ["Weekday pressure compares mean daily observations, not berth-level operations."]
        if "pressure_kind" in work.columns:
            kinds = set(work["pressure_kind"].dropna().astype(str))
            if "full" in kinds and "partial_arrivals_proxy" in kinds:
                work = work[work["pressure_kind"] == "full"].copy()
                caveats.append("Arrivals-only rows were excluded to avoid mixing partial and full pressure indices.")
            elif kinds == {"partial_arrivals_proxy"}:
                caveats.append("This comparison uses the arrivals-only pressure proxy because dwell data is unavailable.")

        work = work.assign(day_of_week=pd.to_datetime(work["date"], errors="coerce", utc=True).dt.day_name())
        pair = (
            work[work["day_of_week"].isin([day_a.title(), day_b.title()])]
            .groupby("day_of_week", dropna=False)
            .agg(congestion_index=("congestion_index", "mean"), observations=("date", "nunique"))
            .reset_index()
        )
        if len(pair) < 2:
            return self.no_data(f"Could not find both weekdays ({day_a.title()}, {day_b.title()}) in pressure history.")
        values = dict(zip(pair["day_of_week"], pair["congestion_index"]))
        a_val = float(values[day_a.title()])
        b_val = float(values[day_b.title()])
        winner = day_a.title() if a_val >= b_val else day_b.title()
        answer = (
            f"{winner} has the higher average port pressure{f' at {port}' if port else ''}: "
            f"{day_a.title()}={a_val:.2f} versus {day_b.title()}={b_val:.2f}."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=pair.sort_values("congestion_index", ascending=False),
            chart=pair.set_index("day_of_week")[["congestion_index"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=caveats,
        )

    def get_pressure_by_vessel_type(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")
        selected = self._filter_dates(self._filter_port(df, port), "date", start, end)
        selected = self._prefer_arrival_source(selected, "date")
        history = self._prefer_arrival_source(self._filter_port(df, port), "date")
        if selected.empty or history.empty:
            return self.no_data("No vessel-type rows matched the requested pressure scope.")

        selected_daily = (
            selected.groupby(["vessel_type_norm", "date"], dropna=False)["arrivals_vessels"].sum().reset_index()
        )
        history_daily = (
            history.groupby(["vessel_type_norm", "date"], dropna=False)["arrivals_vessels"].sum().reset_index()
        )
        selected_daily["day_of_week"] = pd.to_datetime(selected_daily["date"], errors="coerce", utc=True).dt.day_name()
        history_daily["day_of_week"] = pd.to_datetime(history_daily["date"], errors="coerce", utc=True).dt.day_name()
        baseline = (
            history_daily.groupby(["vessel_type_norm", "day_of_week"], dropna=False)["arrivals_vessels"]
            .median()
            .rename("historical_median_daily_arrivals")
            .reset_index()
        )
        scored = selected_daily.merge(baseline, on=["vessel_type_norm", "day_of_week"], how="left")
        scored["pressure_index"] = (
            scored["arrivals_vessels"] / scored["historical_median_daily_arrivals"].replace(0, np.nan)
        ).fillna(0.0).clip(0.0, 5.0)
        table = (
            scored.groupby("vessel_type_norm", dropna=False)
            .agg(
                selected_mean_daily_arrivals=("arrivals_vessels", "mean"),
                selected_arrivals=("arrivals_vessels", "sum"),
                historical_median_daily_arrivals=("historical_median_daily_arrivals", "median"),
                pressure_index=("pressure_index", "mean"),
            )
            .reset_index()
        )
        table["pressure_index"] = table["pressure_index"].clip(0.0, 5.0)
        table["pressure_kind"] = "partial_arrivals_proxy"
        table["pressure_version"] = "pressure_v2"
        table = table.sort_values("pressure_index", ascending=False).reset_index(drop=True)
        leader = table.iloc[0]
        answer = (
            f"{leader['vessel_type_norm']} shows the highest arrivals-based pressure{f' at {port}' if port else ''} "
            f"with an index of {float(leader['pressure_index']):.2f} against its historical daily baseline."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=table,
            chart=table.set_index("vessel_type_norm")[["pressure_index"]],
            coverage_notes=self.coverage_notes(selected, "date"),
            caveats=[
                "Vessel-type pressure is a pressure_v2 arrivals-only proxy because dwell coverage is not segmented by vessel type.",
                "Do not compare this proxy with the full arrivals-plus-dwell pressure index.",
            ],
        )

    def compare_ports(
        self,
        ports: Sequence[str],
        metric: str,
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        source_scope: Optional[str] = None,
    ) -> AnalyticsResult:
        if len(ports) < 2:
            return self.no_data("Comparison needs at least two ports in the question.")

        metric_norm = metric.lower()
        rows: List[Dict[str, Any]] = []
        missing_ports: List[str] = []

        for port in ports:
            if "dwell" in metric_norm:
                result = self.get_avg_dwell_time(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow)
                if result.status != "ok" or result.table is None:
                    missing_ports.append(port)
                    continue
                value = float(result.table["median_dwell_minutes"].median())
                rows.append({"port": port, "metric": "median_dwell_minutes", "value": value})
            elif "congestion" in metric_norm:
                result = self.get_congestion(port=port, start=start, end=end, dow=dow)
                if result.status != "ok" or result.table is None:
                    missing_ports.append(port)
                    continue
                value = float(result.table["congestion_index"].mean())
                kind = (
                    str(result.table["pressure_kind"].dropna().iloc[0])
                    if "pressure_kind" in result.table.columns and result.table["pressure_kind"].notna().any()
                    else "legacy"
                )
                rows.append({"port": port, "metric": "congestion_index", "value": value, "pressure_kind": kind})
            else:
                result = self.get_arrivals(
                    port=port,
                    start=start,
                    end=end,
                    vessel_type=vessel_type,
                    dow=dow,
                    source_scope=source_scope,
                )
                if result.status != "ok" or result.table is None:
                    missing_ports.append(port)
                    continue
                value = float(result.table["arrival_count"].sum())
                rows.append({"port": port, "metric": "arrival_count", "value": value})

        if not rows:
            return self.no_data("No comparable metrics were available for the requested ports.")

        comp = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
        leader = comp.iloc[0]
        metric_name = str(leader["metric"])
        if metric_name == "arrival_count":
            comp["arrival_count"] = pd.to_numeric(comp["value"], errors="coerce")
        if metric_name == "congestion_index" and "pressure_kind" in comp.columns:
            pressure_kinds = set(comp["pressure_kind"].dropna().astype(str))
            if len(pressure_kinds) > 1:
                return self.unsupported(
                    "Requested ports use incompatible pressure calculations, so no cross-scope ranking was produced."
                )
        if metric_name == "arrival_count":
            if len(comp) == 2:
                runner_up = comp.iloc[1]
                lead_value = int(round(float(leader["value"])))
                runner_value = int(round(float(runner_up["value"])))
                answer = (
                    f"{leader['port']} recorded {lead_value} arrivals versus "
                    f"{runner_up['port']} with {runner_value}; {leader['port']} was higher by "
                    f"{lead_value - runner_value} arrivals."
                )
            else:
                values = ", ".join(
                    f"{row.port}={int(round(float(row.value)))}"
                    for row in comp.itertuples(index=False)
                )
                answer = f"Arrival comparison: {values}. {leader['port']} recorded the most arrivals."
        elif metric_name == "congestion_index":
            values = ", ".join(
                f"{row.port}={float(row.value):.2f}"
                for row in comp.itertuples(index=False)
            )
            answer = f"Average port pressure comparison: {values}. {leader['port']} had the highest pressure."
        else:
            values = ", ".join(
                f"{row.port}={float(row.value):.1f} minutes"
                for row in comp.itertuples(index=False)
            )
            answer = f"Median dwell-time comparison: {values}. {leader['port']} had the longest dwell time."
        if missing_ports:
            answer = f"{answer} No matching rows were available for {', '.join(missing_ports)}."
        coverage = [f"Ports compared: {', '.join(ports)}"]
        if missing_ports:
            coverage.append(f"Requested ports without matching rows: {', '.join(missing_ports)}")
        return AnalyticsResult(
            status="partial" if missing_ports else "ok",
            answer=answer,
            table=comp,
            chart=comp.set_index("port")[["arrival_count"]]
            if metric_name == "arrival_count"
            else comp.set_index("port")[["value"]],
            coverage_notes=coverage,
            caveats=[],
        )

    def get_arrivals_multi(
        self,
        ports: Sequence[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        window: Optional[str] = None,
        source_scope: Optional[str] = None,
    ) -> AnalyticsResult:
        clean_ports = [str(p).strip() for p in ports if str(p).strip()]
        if len(clean_ports) < 2:
            return self.no_data("Multi-port aggregate needs at least two ports.")

        rows: List[Dict[str, Any]] = []
        missing_ports: List[str] = []
        for port in clean_ports:
            result = self.get_arrivals(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                dow=dow,
                window=window,
                source_scope=source_scope,
            )
            if result.status != "ok" or result.table is None:
                missing_ports.append(port)
                continue
            rows.append(
                {
                    "port": port,
                    "arrival_count": float(result.table["arrival_count"].sum()),
                    "day_buckets": int(len(result.table)),
                }
            )

        if not rows:
            return self.no_data("No arrival rows matched the requested multi-port scope.")

        table = pd.DataFrame(rows).sort_values("arrival_count", ascending=False).reset_index(drop=True)
        total = int(table["arrival_count"].sum())
        answer = (
            f"Combined arrivals across {len(table)} ports = {total:,} vessel arrivals "
            f"for the selected window ({', '.join(table['port'].tolist())})."
        )
        if missing_ports:
            answer = (
                f"{answer} No matching rows were available for "
                f"{', '.join(missing_ports)}, so the total excludes those ports."
            )
        coverage = [f"Ports aggregated: {', '.join(table['port'].tolist())}"]
        if missing_ports:
            coverage.append(f"Requested ports without matching rows: {', '.join(missing_ports)}")
        return AnalyticsResult(
            status="partial" if missing_ports else "ok",
            answer=answer,
            table=table,
            chart=table.set_index("port")[["arrival_count"]],
            coverage_notes=coverage,
            caveats=["Combined total is the sum of per-port arrival counts in the same date window."],
        )

    def get_first_arrival(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        if self.dwell.empty:
            return self.no_data("dwell_time.parquet is missing.")
        if not port:
            return self.no_data("First-arrival query requires a target port.")

        work = self._filter_port(self.dwell, port)
        work = self._filter_dates(work, "arrival_date", start, end, window=window)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No arrival rows matched this port/date scope.")

        if "arrival_time" in work.columns:
            work = work.dropna(subset=["arrival_time"]).sort_values("arrival_time")
        elif "arrival_date" in work.columns:
            work = work.dropna(subset=["arrival_date"]).sort_values("arrival_date")
        else:
            return self.no_data("No arrival timestamp fields are available for first-arrival analysis.")
        if work.empty:
            return self.no_data("No valid arrival timestamps were available for this scope.")

        first = work.iloc[0]
        arrival_ts = pd.to_datetime(first.get("arrival_time") or first.get("arrival_date"), errors="coerce", utc=True)
        arrival_label = arrival_ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(arrival_ts) else "unknown time"
        mmsi = str(first.get("mmsi") or "unknown")
        answer = f"First recorded arrival at {port} in this window is MMSI {mmsi} at {arrival_label}."

        cols = [c for c in ["mmsi", "port_key", "port_label", "arrival_time", "departure_time", "dwell_minutes", "vessel_type_norm"] if c in work.columns]
        table = work[cols].head(10).reset_index(drop=True)
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=table,
            chart=None,
            coverage_notes=self.coverage_notes(work, "arrival_date"),
            caveats=["First-arrival uses the earliest available arrival timestamp in filtered port-call dwell rows."],
        )

    def get_last_arrival(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        if self.dwell.empty:
            return self.no_data("dwell_time.parquet is missing.")
        if not port:
            return self.no_data("Last-arrival query requires a target port.")

        work = self._filter_port(self.dwell, port)
        work = self._filter_dates(work, "arrival_date", start, end, window=window)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No arrival rows matched this port/date scope.")

        if "arrival_time" in work.columns:
            work = work.dropna(subset=["arrival_time"]).sort_values("arrival_time", ascending=False)
        elif "arrival_date" in work.columns:
            work = work.dropna(subset=["arrival_date"]).sort_values("arrival_date", ascending=False)
        else:
            return self.no_data("No arrival timestamp fields are available for last-arrival analysis.")
        if work.empty:
            return self.no_data("No valid arrival timestamps were available for this scope.")

        last = work.iloc[0]
        arrival_ts = pd.to_datetime(last.get("arrival_time") or last.get("arrival_date"), errors="coerce", utc=True)
        arrival_label = arrival_ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(arrival_ts) else "unknown time"
        mmsi = str(last.get("mmsi") or "unknown")
        answer = f"Last recorded arrival at {port} in this window is MMSI {mmsi} at {arrival_label}."

        cols = [c for c in ["mmsi", "port_key", "port_label", "arrival_time", "departure_time", "dwell_minutes", "vessel_type_norm"] if c in work.columns]
        table = work[cols].head(10).reset_index(drop=True)
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=table,
            chart=None,
            coverage_notes=self.coverage_notes(work, "arrival_date"),
            caveats=["Last-arrival uses the latest available arrival timestamp in filtered port-call dwell rows."],
        )

    def get_first_departure(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        if self.dwell.empty:
            return self.no_data("dwell_time.parquet is missing.")
        if not port:
            return self.no_data("First-departure query requires a target port.")

        work = self._filter_port(self.dwell, port)
        work = self._filter_dates(work, "arrival_date", start, end, window=window)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No departure rows matched this port/date scope.")
        if "departure_time" not in work.columns:
            return self.no_data("Departure timestamps are not available for this scope.")

        work = work.dropna(subset=["departure_time"]).sort_values("departure_time")
        if work.empty:
            return self.no_data("No valid departure timestamps were available for this scope.")

        first = work.iloc[0]
        dep_ts = pd.to_datetime(first.get("departure_time"), errors="coerce", utc=True)
        dep_label = dep_ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(dep_ts) else "unknown time"
        mmsi = str(first.get("mmsi") or "unknown")
        answer = f"First recorded departure from {port} in this window is MMSI {mmsi} at {dep_label}."

        cols = [c for c in ["mmsi", "port_key", "port_label", "arrival_time", "departure_time", "dwell_minutes", "vessel_type_norm"] if c in work.columns]
        table = work[cols].head(10).reset_index(drop=True)
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=table,
            chart=None,
            coverage_notes=self.coverage_notes(work, "arrival_date"),
            caveats=["First-departure uses the earliest available departure timestamp in filtered port-call dwell rows."],
        )

    def get_first_route_vessel(
        self,
        origin_port: Optional[str],
        destination_port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.voyages
        if df.empty:
            return self.no_data("voyages.parquet is missing.")
        if not origin_port or not destination_port:
            return self.no_data("Route-first query requires both origin and destination ports.")

        work = self._filter_voyage_endpoint(df, "origin", origin_port)
        work = self._filter_voyage_endpoint(work, "destination", destination_port)
        work = self._filter_dates(work, "departure_time", start, end, window=window)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No voyage rows matched this origin→destination window.")

        if "departure_time" not in work.columns:
            return self.no_data("Voyage departure timestamps are missing for route-first analysis.")
        work = work.dropna(subset=["departure_time"]).sort_values("departure_time")
        if work.empty:
            return self.no_data("No valid departure timestamps were available for this route scope.")

        first = work.iloc[0]
        dep_ts = pd.to_datetime(first.get("departure_time"), errors="coerce", utc=True)
        arr_ts = pd.to_datetime(first.get("arrival_time"), errors="coerce", utc=True)
        dep_label = dep_ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(dep_ts) else "unknown"
        arr_label = arr_ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(arr_ts) else "unknown"
        mmsi = str(first.get("mmsi") or "unknown")
        voyage_id = str(first.get("voyage_id") or "")
        duration_h = float(pd.to_numeric(first.get("duration_h"), errors="coerce") or 0.0)
        answer = (
            f"First recorded vessel on {origin_port}→{destination_port} in this window is MMSI {mmsi} "
            f"(voyage {voyage_id}) departing {dep_label}, arriving {arr_label}, duration {duration_h:.2f} h."
        )

        cols = [
            c
            for c in [
                "voyage_id",
                "mmsi",
                "imo",
                "origin_port_key",
                "destination_port_key",
                "departure_time",
                "arrival_time",
                "duration_h",
                "vessel_type_norm",
                "source_kind",
            ]
            if c in work.columns
        ]
        table = work[cols].head(10).reset_index(drop=True)
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=table,
            chart=None,
            coverage_notes=self.coverage_notes(work, "departure_time"),
            caveats=["Route-first uses earliest departure_time from reconstructed voyage rows for the selected OD pair."],
        )

    def get_route_travel_time_summary(
        self,
        origin_port: Optional[str],
        destination_port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.voyages
        if df.empty:
            return self.no_data("voyages.parquet is missing.")
        if not origin_port or not destination_port:
            return self.no_data("Route travel-time summary requires both origin and destination ports.")

        work = self._filter_voyage_endpoint(df, "origin", origin_port)
        work = self._filter_voyage_endpoint(work, "destination", destination_port)
        work = self._filter_dates(work, "departure_time", start, end, window=window)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No voyage rows matched this origin→destination window.")

        work = work.copy()
        work["duration_h"] = pd.to_numeric(work.get("duration_h"), errors="coerce")
        work = work.dropna(subset=["duration_h"])
        if work.empty:
            return self.no_data("No route-duration values were available for this route scope.")

        median_h = float(work["duration_h"].median())
        p90_h = float(work["duration_h"].quantile(0.90))
        n = int(len(work))
        answer = (
            f"Route travel time for {origin_port}→{destination_port}: "
            f"median={median_h:.2f} h, p90={p90_h:.2f} h over {n:,} voyage(s)."
        )

        out = (
            work.sort_values("duration_h", ascending=False)[
                [
                    c
                    for c in [
                        "voyage_id",
                        "mmsi",
                        "origin_port_key",
                        "destination_port_key",
                        "departure_time",
                        "arrival_time",
                        "duration_h",
                        "vessel_type_norm",
                    ]
                    if c in work.columns
                ]
            ]
            .head(20)
            .reset_index(drop=True)
        )
        # The visual contract for a route-duration summary is a distribution or
        # percentile view.  A daily-median trend is a different question and can
        # hide the long tail, so expose the exact headline percentiles instead.
        chart = pd.DataFrame(
            [
                {"percentile": "p50", "duration_h": median_h},
                {"percentile": "p90", "duration_h": p90_h},
            ]
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=out,
            chart=chart if not chart.empty else None,
            coverage_notes=self.coverage_notes(work, "departure_time"),
            caveats=["Route-duration summary is based on reconstructed voyage episodes in the filtered OD/date scope."],
        )

    def compare_ports_and_routes(
        self,
        ports: Sequence[str],
        route_pairs: Sequence[Dict[str, str]],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        rows: List[Dict[str, Any]] = []
        resolved_ports: List[str] = []
        missing_ports: List[str] = []
        resolved_routes: List[str] = []
        missing_routes: List[str] = []

        for port in [str(p).strip() for p in ports if str(p).strip()]:
            result = self.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window)
            if result.status != "ok" or result.table is None:
                missing_ports.append(port)
                continue
            resolved_ports.append(port)
            rows.append(
                {
                    "scope_type": "port",
                    "scope_label": port,
                    "metric": "arrival_count",
                    "value": float(result.table["arrival_count"].sum()),
                }
            )

        for pair in route_pairs:
            origin = str(pair.get("origin") or "").strip()
            destination = str(pair.get("destination") or "").strip()
            if not origin or not destination:
                continue
            result = self.get_route_travel_time_summary(
                origin_port=origin,
                destination_port=destination,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
            if result.status != "ok" or result.table is None:
                missing_routes.append(f"{origin}->{destination}")
                continue
            work = self._filter_voyage_endpoint(self.voyages, "origin", origin)
            work = self._filter_voyage_endpoint(work, "destination", destination)
            work = self._filter_dates(work, "departure_time", start, end, window=window)
            work = self._filter_vessel_type(work, vessel_type)
            work = work.copy()
            work["duration_h"] = pd.to_numeric(work.get("duration_h"), errors="coerce")
            work = work.dropna(subset=["duration_h"])
            if work.empty:
                missing_routes.append(f"{origin}->{destination}")
                continue
            resolved_routes.append(f"{origin}->{destination}")
            rows.append(
                {
                    "scope_type": "route",
                    "scope_label": f"{origin}->{destination}",
                    "metric": "route_duration_median_h",
                    "value": float(work["duration_h"].median()),
                }
            )
            rows.append(
                {
                    "scope_type": "route",
                    "scope_label": f"{origin}->{destination}",
                    "metric": "route_duration_p90_h",
                    "value": float(work["duration_h"].quantile(0.90)),
                }
            )

        if not rows:
            return self.no_data("No comparable port/route metrics were available for this combined prompt.")

        table = pd.DataFrame(rows).sort_values(["scope_type", "scope_label", "metric"]).reset_index(drop=True)
        port_labels = sorted({str(p) for p in ports if str(p).strip()})
        route_labels = sorted(
            {
                f"{str(p.get('origin') or '').strip()}->{str(p.get('destination') or '').strip()}"
                for p in route_pairs
                if str(p.get("origin") or "").strip() and str(p.get("destination") or "").strip()
            }
        )
        computed_summary = (
            f"Computed {len(set(resolved_ports))} requested port scope(s) "
            f"and {len(set(resolved_routes))} requested route scope(s)."
        )
        unavailable: List[str] = []
        if missing_ports:
            unavailable.append(f"ports: {', '.join(sorted(set(missing_ports)))}")
        if missing_routes:
            unavailable.append(f"routes: {', '.join(sorted(set(missing_routes)))}")
        answer = computed_summary
        if unavailable:
            answer = (
                f"{computed_summary} No matching rows were available for "
                f"{'; '.join(unavailable)}."
            )
        chart = None
        if not table.empty:
            chart = table.assign(series=table["scope_type"] + ":" + table["metric"]).set_index("scope_label")[["value"]]
        return AnalyticsResult(
            status="partial" if unavailable else "ok",
            answer=answer,
            table=table,
            chart=chart,
            coverage_notes=[
                f"Ports requested: {', '.join(port_labels) if port_labels else 'none'}",
                f"Ports computed: {', '.join(sorted(set(resolved_ports))) if resolved_ports else 'none'}",
                f"Routes requested: {', '.join(route_labels) if route_labels else 'none'}",
                f"Routes computed: {', '.join(sorted(set(resolved_routes))) if resolved_routes else 'none'}",
            ],
            caveats=[
                "This combined table mixes port arrivals counts and route duration statistics; compare within each metric family.",
            ],
        )

    def diagnose_congestion(
        self,
        port: Optional[str],
        target_date: Optional[str],
    ) -> AnalyticsResult:
        if not target_date:
            return self.no_data("Diagnostic questions need a specific date (YYYY-MM-DD).")

        all_arrivals = self._prefer_arrival_source(self._filter_port(self.arrivals_daily, port), "date")
        arrivals = self._filter_dates(all_arrivals, "date", target_date, target_date)
        if arrivals.empty:
            return self.no_data("No arrivals found for this port/date diagnostic query.")

        arrivals_total = int(arrivals["arrival_count"].sum())
        target_by_type = (
            arrivals.groupby("vessel_type_norm", dropna=False)
            .agg(observed_arrivals=("arrival_count", "sum"))
            .reset_index()
        )

        target_ts = pd.Timestamp(target_date, tz="UTC")
        target_dow = target_ts.day_name()
        history = all_arrivals.copy()
        history_dates = pd.to_datetime(history["date"], errors="coerce", utc=True)
        history = history[(history_dates.dt.day_name() == target_dow) & (history_dates != target_ts)]
        history_daily_type = (
            history.groupby(["date", "vessel_type_norm"], dropna=False)["arrival_count"].sum().reset_index()
        )
        baseline_by_type = (
            history_daily_type.groupby("vessel_type_norm", dropna=False)["arrival_count"]
            .median()
            .rename("weekday_baseline_arrivals")
            .reset_index()
        )
        contributors = target_by_type.merge(baseline_by_type, on="vessel_type_norm", how="outer").fillna(0.0)
        contributors["excess_arrivals"] = (
            contributors["observed_arrivals"] - contributors["weekday_baseline_arrivals"]
        )
        contributors = contributors.sort_values(
            ["excess_arrivals", "observed_arrivals"], ascending=[False, False]
        ).reset_index(drop=True)

        historical_daily = history.groupby("date", dropna=False)["arrival_count"].sum()
        arrivals_baseline = float(historical_daily.median()) if not historical_daily.empty else float("nan")
        arrivals_delta_pct = (
            (float(arrivals_total) / arrivals_baseline - 1.0) * 100.0
            if pd.notna(arrivals_baseline) and arrivals_baseline > 0
            else float("nan")
        )

        dwell_note = " Dwell data was not available for this date."
        if not self.dwell.empty:
            all_dwell = self._filter_port(self.dwell, port)
            dwell = self._filter_dates(all_dwell, "arrival_date", target_date, target_date)
            if not dwell.empty:
                target_dwell = float(dwell["dwell_minutes"].median())
                dwell_dates = pd.to_datetime(all_dwell["arrival_date"], errors="coerce", utc=True)
                dwell_history = all_dwell[(dwell_dates.dt.day_name() == target_dow) & (dwell_dates != target_ts)]
                dwell_daily = dwell_history.groupby("arrival_date", dropna=False)["dwell_minutes"].median()
                baseline_dwell = float(dwell_daily.median()) if not dwell_daily.empty else float("nan")
                dwell_note = (
                    f" Median dwell was {target_dwell:.1f} minutes"
                    + (f" versus a {target_dow} baseline of {baseline_dwell:.1f} minutes" if pd.notna(baseline_dwell) else "")
                    + f" across {len(dwell):,} calls."
                )

        delta_text = (
            f" ({arrivals_delta_pct:+.1f}%)" if pd.notna(arrivals_delta_pct) else ""
        )
        baseline_text = f"{arrivals_baseline:.1f}" if pd.notna(arrivals_baseline) else "unavailable"
        ranked = contributors[contributors["excess_arrivals"] > 0].head(3)
        contributor_text = ", ".join(
            f"{row.vessel_type_norm} ({float(row.excess_arrivals):+.1f} versus baseline)"
            for row in ranked.itertuples(index=False)
        )
        if not contributor_text:
            contributor_text = "no vessel type exceeded its weekday baseline"
        answer = (
            f"On {target_date}, {arrivals_total:,} arrivals were recorded"
            + (f" for {port}" if port else "")
            + f" versus a {target_dow} baseline of {baseline_text}{delta_text}."
            + dwell_note
            + f" The largest observed arrival contributors were {contributor_text}."
        )

        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=contributors,
            chart=contributors.set_index("vessel_type_norm")[["observed_arrivals", "weekday_baseline_arrivals"]],
            coverage_notes=[
                f"Diagnostic date: {target_date}",
                f"Baseline: other {target_dow}s in available historical coverage",
                f"Rows used for target: {len(arrivals):,}",
            ],
            caveats=[
                "Contributors are ranked observed associations against a weekday baseline; they do not establish causation.",
                self._arrivals_source_note(arrivals, None),
            ],
        )

    def detect_arrival_spikes(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        if work.empty:
            return self.no_data("No arrival rows matched these filters.")

        daily = (
            work.groupby("date", dropna=False)
            .agg(arrival_count=("arrival_count", "sum"))
            .reset_index()
            .sort_values("date")
            .reset_index(drop=True)
        )
        daily["roll_mean_7"] = daily["arrival_count"].rolling(7, min_periods=3).mean().shift(1)
        daily["roll_std_7"] = daily["arrival_count"].rolling(7, min_periods=3).std().shift(1)
        daily["threshold"] = daily["roll_mean_7"] + 2.0 * daily["roll_std_7"].fillna(0)
        daily["is_anomaly"] = daily["arrival_count"] > daily["threshold"]
        spikes = daily[daily["is_anomaly"]].copy()

        if spikes.empty:
            return AnalyticsResult(
                status="ok",
                answer="No statistically unusual arrival spikes were detected in the selected period.",
                table=daily.tail(20),
                chart=daily.set_index("date")[["arrival_count", "threshold", "is_anomaly"]],
                coverage_notes=self.coverage_notes(work, "date"),
                caveats=["Spike rule: arrivals > rolling_mean_7 + 2*rolling_std_7."],
            )

        answer = f"Detected {len(spikes)} potential arrival spike days." 
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=spikes[["date", "arrival_count", "threshold", "is_anomaly"]],
            chart=daily.set_index("date")[["arrival_count", "threshold", "is_anomaly"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=["Spike rule: arrivals > rolling_mean_7 + 2*rolling_std_7."],
        )
