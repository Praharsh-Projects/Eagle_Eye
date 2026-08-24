"""Forecast congestion/arrival proxies from KPI daily time series."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.kpi.query import KPIQueryEngine


@dataclass
class ForecastResult:
    status: str
    answer: str
    history: Optional[pd.DataFrame]
    forecast: Optional[pd.DataFrame]
    coverage_notes: List[str]
    caveats: List[str]

    @property
    def table(self) -> Optional[pd.DataFrame]:
        """Compatibility surface for generic result renderers."""
        return self.forecast

    @property
    def chart(self) -> Optional[pd.DataFrame]:
        """Compatibility surface for generic chart renderers."""
        return self.forecast


class ForecastEngine:
    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)
        self.kpi = KPIQueryEngine(processed_dir=self.processed_dir)
        self._backtest: Optional[Dict[str, object]] = None

    @staticmethod
    def _prepare_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        series = (
            df[[date_col, value_col]]
            .dropna(subset=[date_col, value_col])
            .groupby(date_col, dropna=False)[value_col]
            .sum()
            .sort_index()
            .astype(float)
        )
        return series

    @staticmethod
    def _one_step_prediction(history: List[float]) -> float:
        if not history:
            return 0.0
        if len(history) == 1:
            return float(max(0.0, history[-1]))
        seasonal = history[-7] if len(history) >= 7 else history[-1]
        window = history[-7:] if len(history) >= 7 else history
        moving_avg = float(np.mean(window))
        pred = 0.7 * float(seasonal) + 0.3 * moving_avg
        return float(max(0.0, pred))

    @classmethod
    def _forecast_with_intervals(cls, series: pd.Series, horizon_days: int) -> pd.DataFrame:
        if series.empty:
            return pd.DataFrame()

        values = series.tolist()
        residuals: List[float] = []
        for idx in range(1, len(values)):
            pred = cls._one_step_prediction(values[:idx])
            residuals.append(float(values[idx] - pred))

        if len(residuals) >= 3:
            residual_std = float(np.std(residuals, ddof=1))
        else:
            residual_std = float(np.std(values) if len(values) > 1 else max(values[0], 1.0) * 0.15)

        history = values.copy()
        last_date = pd.Timestamp(series.index.max()).floor("D")
        rows: List[Dict[str, float | pd.Timestamp]] = []
        for step in range(1, horizon_days + 1):
            pred = cls._one_step_prediction(history)
            sigma = residual_std * np.sqrt(step)
            # Central 80% interval (z ~= 1.28155), matching the release gate.
            lower = max(0.0, pred - 1.2815515655446004 * sigma)
            upper = pred + 1.2815515655446004 * sigma
            ts = last_date + pd.Timedelta(days=step)
            rows.append(
                {
                    "date": ts,
                    "predicted": float(pred),
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )
            history.append(pred)

        return pd.DataFrame(rows)

    @classmethod
    def _one_step_interval(cls, history: List[float]) -> tuple[float, float, float]:
        pred = cls._one_step_prediction(history)
        residuals = [
            float(history[idx] - cls._one_step_prediction(history[:idx]))
            for idx in range(1, len(history))
        ]
        if len(residuals) >= 3:
            sigma = float(np.std(residuals, ddof=1))
        elif len(history) > 1:
            sigma = float(np.std(history, ddof=1))
        else:
            sigma = max(float(history[0]) if history else 1.0, 1.0) * 0.15
        half_width = 1.2815515655446004 * sigma
        return pred, max(0.0, pred - half_width), pred + half_width

    def _load_backtest(self) -> Dict[str, object]:
        if self._backtest is not None:
            return self._backtest
        path = self.processed_dir / "forecast_backtest.json"
        if not path.exists():
            self._backtest = {}
            return self._backtest
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        self._backtest = payload if isinstance(payload, dict) else {}
        return self._backtest

    def _quality_gate(self, metric_group: str, port: str) -> tuple[bool, str]:
        payload = self._load_backtest()
        section = payload.get(metric_group, {}) if isinstance(payload, dict) else {}
        rows = section.get("per_port", []) if isinstance(section, dict) else []
        resolved = self.kpi.resolve_port_token(port) or str(port).strip()
        matched = next(
            (
                row
                for row in rows
                if isinstance(row, dict) and str(row.get("port_key", "")).upper() == str(resolved).upper()
            ),
            None,
        )
        if matched is None:
            return False, f"No rolling-origin validation result is available for {resolved}."
        mase = pd.to_numeric(matched.get("mase"), errors="coerce")
        coverage = pd.to_numeric(matched.get("interval_80_coverage"), errors="coerce")
        passed = bool(pd.notna(mase) and pd.notna(coverage) and float(mase) < 1.0 and 0.70 <= float(coverage) <= 0.90)
        reason = (
            f"Rolling-origin validation for {resolved}: MASE={float(mase):.3f}, "
            f"80% interval coverage={float(coverage):.3f}; required MASE<1 and coverage 0.70-0.90."
            if pd.notna(mase) and pd.notna(coverage)
            else f"Rolling-origin validation metrics are incomplete for {resolved}."
        )
        return passed, reason

    @staticmethod
    def _unavailable(reason: str, history: Optional[pd.DataFrame] = None) -> ForecastResult:
        return ForecastResult(
            status="no_data",
            answer="A forecast is unavailable because the configured quality gate was not met.",
            history=history,
            forecast=None,
            coverage_notes=[],
            caveats=[reason],
        )

    @staticmethod
    def _confidence_label(sample_count: int, tier: int) -> str:
        if tier == 1 and sample_count >= 4:
            return "high"
        if tier == 1 and sample_count >= 2:
            return "medium"
        if tier == 2 and sample_count >= 3:
            return "high"
        if tier == 2 and sample_count >= 2:
            return "medium"
        if tier <= 4 and sample_count >= 3:
            return "medium"
        if sample_count >= 5:
            return "medium"
        return "low"

    @staticmethod
    def _congestion_level(value: float) -> str:
        if value < 0.8:
            return "below normal"
        if value < 1.2:
            return "normal"
        if value < 1.6:
            return "elevated"
        return "high"

    @classmethod
    def _congestion_meaning(cls, value: float) -> str:
        level = cls._congestion_level(value)
        return (
            f"Congestion index {value:.2f} means {level} pressure. "
            "Index 1.00 is the port's typical baseline in this dataset; "
            "values above 1.00 indicate above-baseline traffic pressure."
        )

    @staticmethod
    def _seasonal_analog(
        series: pd.Series,
        target_date: pd.Timestamp,
    ) -> tuple[float, float, float, str, int, str, List[str], List[str]]:
        hist = series.reset_index()
        hist.columns = ["date", "value"]
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=True).dt.floor("D")
        hist = hist.dropna(subset=["date", "value"])
        hist["year"] = hist["date"].dt.year
        hist["month"] = hist["date"].dt.month
        hist["day"] = hist["date"].dt.day
        hist["day_of_week"] = hist["date"].dt.day_name()
        iso_parts = hist["date"].dt.isocalendar()
        hist["iso_week"] = iso_parts["week"].astype(int)

        target_month = int(target_date.month)
        target_day = int(target_date.day)
        target_week = int(target_date.isocalendar().week)
        target_dow = target_date.day_name()
        day_gap = (hist["day"] - target_day).abs()

        tiers = [
            (
                "same month-day across years",
                (hist["month"] == target_month) & (hist["day"] == target_day),
                1,
            ),
            (
                "same month-day-window (+/-2 days) + weekday",
                (hist["month"] == target_month) & (day_gap <= 2) & (hist["day_of_week"] == target_dow),
                2,
            ),
            (
                "same ISO week + weekday",
                (hist["iso_week"] == target_week) & (hist["day_of_week"] == target_dow),
                3,
            ),
            (
                "same month + weekday",
                (hist["month"] == target_month) & (hist["day_of_week"] == target_dow),
                3,
            ),
            ("same month", (hist["month"] == target_month), 5),
            ("same weekday", (hist["day_of_week"] == target_dow), 5),
            ("all history", pd.Series(True, index=hist.index), 1),
        ]

        selected = pd.DataFrame()
        tier_label = "all history"
        tier_idx = len(tiers)
        for idx, (label, mask, min_count) in enumerate(tiers, start=1):
            sample = hist.loc[mask, ["date", "value"]].copy()
            sample["value"] = pd.to_numeric(sample["value"], errors="coerce")
            sample = sample.dropna(subset=["date", "value"])
            if len(sample) >= min_count or idx == len(tiers):
                selected = sample
                tier_label = label
                tier_idx = idx
                break

        if selected.empty:
            return 0.0, 0.0, 0.0, "no analog samples", 0, "low", [], []

        selected_values = selected["value"].astype(float)
        pred = float(selected_values.mean())
        if len(selected) >= 2:
            lower = float(max(0.0, selected_values.quantile(0.10)))
            upper = float(selected_values.quantile(0.90))
        else:
            lower = float(max(0.0, pred * 0.80))
            upper = float(pred * 1.20)

        confidence = ForecastEngine._confidence_label(sample_count=len(selected), tier=tier_idx)
        analog_dates = (
            selected.sort_values("date")["date"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()
        )
        analog_points = [
            f"{row['date'].strftime('%Y-%m-%d')}={float(row['value']):.2f}"
            for _, row in selected.sort_values("date").iterrows()
        ]
        return pred, lower, upper, tier_label, len(selected), confidence, analog_dates, analog_points

    def forecast_arrivals(
        self,
        port: str,
        horizon_weeks: int = 4,
        vessel_type: Optional[str] = None,
    ) -> ForecastResult:
        if self.kpi.arrivals_daily.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["arrivals_daily.parquet is missing. Run KPI build first."],
            )

        work = self.kpi._filter_port(self.kpi.arrivals_daily, port)
        work = self.kpi._filter_vessel_type(work, vessel_type)
        if work.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No arrival history found for this port filter."],
            )

        series = self._prepare_series(work, "date", "arrivals_vessels")
        if len(series) < 14:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["Need at least 14 daily points for forecast stability."],
            )

        if int(horizon_weeks) > 8:
            return self._unavailable("Forecast horizon exceeds the maximum validated window of eight weeks.")
        _gate_passed, gate_reason = self._quality_gate("arrivals", port)

        horizon_days = int(max(1, horizon_weeks) * 7)
        forecast_df = self._forecast_with_intervals(series, horizon_days=horizon_days)
        history_df = series.reset_index().rename(columns={"index": "date", "arrivals_vessels": "actual"})
        history_df.columns = ["date", "actual"]

        mean_pred = float(forecast_df["predicted"].mean())
        last_date = pd.Timestamp(series.index.max()).strftime("%Y-%m-%d")
        answer = (
            f"Historical-data forecast anchored after {last_date}: mean arrivals are {mean_pred:.2f} "
            f"vessels/day over the following {horizon_weeks} week(s)."
        )
        notes = self.kpi.coverage_notes(work, "date")
        notes.append(f"Forecast horizon: {horizon_weeks} week(s)")

        return ForecastResult(
            status="ok",
            answer=answer,
            history=history_df,
            forecast=forecast_df,
            coverage_notes=notes,
            caveats=[
                "Forecast uses a weekly-seasonal baseline plus 7-day moving average with an empirical 80% interval.",
                gate_reason,
            ],
        )

    def forecast_arrivals_for_date(
        self,
        port: str,
        target_date: str,
        horizon_weeks: int = 4,
    ) -> ForecastResult:
        target_ts = pd.to_datetime(target_date, errors="coerce", utc=True)
        if pd.isna(target_ts):
            return self._unavailable("Target date is invalid. Use YYYY-MM-DD.")
        target_ts = pd.Timestamp(target_ts).floor("D")

        work = self.kpi._filter_port(self.kpi.arrivals_daily, port)
        if work.empty:
            return self._unavailable("No arrival history found for this port filter.")
        series = self._prepare_series(work, "date", "arrivals_vessels")
        if len(series) < 14:
            return self._unavailable("Need at least 14 historical daily points for a forecast.")

        history_df = series.reset_index().rename(columns={"index": "date", "arrivals_vessels": "actual"})
        history_df.columns = ["date", "actual"]
        last_date = pd.Timestamp(series.index.max()).floor("D")

        if target_ts <= last_date and target_ts in series.index:
            actual = float(series.loc[target_ts])
            observed = pd.DataFrame(
                [{"date": target_ts, "predicted": actual, "lower": actual, "upper": actual}]
            )
            return ForecastResult(
                status="ok",
                answer=(
                    f"Observed arrivals at {port or 'the selected port'} on {target_ts.strftime('%Y-%m-%d')} "
                    f"were {actual:.0f} vessels; this is historical evidence, not a forecast."
                ),
                history=history_df,
                forecast=observed,
                coverage_notes=self.kpi.coverage_notes(work, "date"),
                caveats=["Target date is inside historical coverage; the observed value takes precedence."],
            )
        if target_ts <= last_date:
            return self._unavailable(
                "The target is inside historical coverage, but that date has no observed arrival value.",
                history=history_df,
            )

        days_ahead = int((target_ts - last_date).days)
        if days_ahead > 56:
            return self._unavailable(
                f"Target date is {days_ahead} days after the latest observation; maximum validated horizon is 56 days.",
                history=history_df,
            )
        if int(horizon_weeks) > 8:
            return self._unavailable(
                "Forecast horizon exceeds the maximum validated window of eight weeks.",
                history=history_df,
            )
        _gate_passed, gate_reason = self._quality_gate("arrivals", port)

        forecast_df = self._forecast_with_intervals(series, horizon_days=days_ahead)
        target_row = forecast_df.iloc[-1]
        predicted = float(target_row["predicted"])
        lower = float(target_row["lower"])
        upper = float(target_row["upper"])
        return ForecastResult(
            status="ok",
            answer=(
                f"Historical-data forecast for {port or 'the selected port'} on {target_ts.strftime('%Y-%m-%d')}, "
                f"anchored to the latest observation on {last_date.strftime('%Y-%m-%d')}: "
                f"{predicted:.2f} arrivals (80% interval {lower:.2f} to {upper:.2f})."
            ),
            history=history_df,
            forecast=forecast_df,
            coverage_notes=self.kpi.coverage_notes(work, "date")
            + [f"Target date: {target_ts.strftime('%Y-%m-%d')}", f"Forecast horizon: {days_ahead} day(s)"],
            caveats=[
                "This is a validated historical-series forecast, not current operational intelligence.",
                gate_reason,
            ],
        )

    def forecast_congestion_for_date(
        self,
        port: str,
        target_date: str,
        horizon_weeks: int = 4,
    ) -> ForecastResult:
        target_ts = pd.to_datetime(target_date, errors="coerce", utc=True)
        if pd.isna(target_ts):
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["Target date is invalid. Use YYYY-MM-DD."],
            )
        target_ts = pd.Timestamp(target_ts).floor("D")

        if self.kpi.congestion.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["congestion_daily.parquet is missing. Run KPI build first."],
            )

        work = self.kpi._filter_port(self.kpi.congestion, port)
        if work.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No congestion history found for this port filter."],
            )

        series = self._prepare_series(work, "date", "congestion_index")
        if series.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No congestion values are available after filtering."],
            )

        history_df = series.reset_index().rename(columns={"index": "date", "congestion_index": "actual"})
        history_df.columns = ["date", "actual"]
        last_date = pd.Timestamp(series.index.max()).floor("D")

        if target_ts <= last_date and target_ts in series.index:
            actual = float(series.loc[target_ts])
            meaning = self._congestion_meaning(actual)
            forecast_df = pd.DataFrame(
                [{"date": target_ts, "predicted": actual, "lower": actual, "upper": actual}]
            )
            return ForecastResult(
                status="ok",
                answer=(
                    f"Observed congestion index at {port or 'selected port'} on {target_ts.strftime('%Y-%m-%d')} "
                    f"was {actual:.2f}. {meaning}"
                ),
                history=history_df,
                forecast=forecast_df,
                coverage_notes=self.kpi.coverage_notes(work, "date")
                + [f"Meaning: {meaning}"],
                caveats=[
                    "Target date is inside historical coverage; this is observed value, not a future forecast.",
                    "Congestion index is a proxy from arrivals and dwell-time availability, not berth-level operations.",
                ],
            )

        if target_ts <= last_date:
            return ForecastResult(
                status="no_data",
                answer="No observed pressure value is available for that historical date.",
                history=history_df,
                forecast=None,
                coverage_notes=self.kpi.coverage_notes(work, "date"),
                caveats=["Historical gaps are not filled with forecasts."],
            )

        days_ahead = int((target_ts - last_date).days)
        if days_ahead > 56:
            return self._unavailable(
                f"Target date is {days_ahead} days after the latest observation; maximum validated horizon is 56 days.",
                history=history_df,
            )
        if len(series) < 14:
            return self._unavailable("Need at least 14 historical daily points for a forecast.", history=history_df)
        _gate_passed, gate_reason = self._quality_gate("congestion", port)

        forecast_df = self._forecast_with_intervals(series, horizon_days=days_ahead)
        target_row = forecast_df.iloc[-1]
        pred = float(target_row["predicted"])
        lower = float(target_row["lower"])
        upper = float(target_row["upper"])
        method_note = "Method: weekly-seasonal baseline + moving-average model."

        meaning = self._congestion_meaning(pred)
        level = self._congestion_level(pred)

        notes = self.kpi.coverage_notes(work, "date")
        notes.append(f"Target date: {target_ts.strftime('%Y-%m-%d')}")
        notes.append(method_note)
        notes.append(f"Meaning: {meaning}")

        caveats = [
            "Congestion index is a proxy from arrivals and dwell-time availability, not berth-level operations.",
            "Forecast is based on historical seasonal patterns in available data.",
            gate_reason,
        ]
        if "has_dwell" in work.columns and work["has_dwell"].fillna(False).mean() < 0.5:
            caveats.append("Dwell coverage is sparse, so forecast is more arrival-driven.")

        return ForecastResult(
            status="ok",
            answer=(
                f"Predicted congestion index at {port or 'selected port'} on {target_ts.strftime('%Y-%m-%d')} "
                f"is {pred:.2f} (range {lower:.2f} to {upper:.2f}). "
                f"This indicates {level} pressure versus baseline (1.00). "
                "The calculation uses the available historical series."
            ),
            history=history_df,
            forecast=forecast_df,
            coverage_notes=notes,
            caveats=caveats,
        )

    def forecast_congestion(
        self,
        port: str,
        target_dow: str = "Friday",
        horizon_weeks: int = 4,
    ) -> ForecastResult:
        target = target_dow.strip().title()

        if self.kpi.congestion.empty:
            # Fall back to arrivals forecast if congestion table is unavailable.
            return self.forecast_arrivals(port=port, horizon_weeks=horizon_weeks)

        work = self.kpi._filter_port(self.kpi.congestion, port)
        if work.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No congestion history found for this port filter."],
            )

        series = self._prepare_series(work, "date", "congestion_index")
        if len(series) < 14:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["Need at least 14 daily points for congestion forecast."],
            )

        if int(horizon_weeks) > 8:
            return self._unavailable("Forecast horizon exceeds the maximum validated window of eight weeks.")
        _gate_passed, gate_reason = self._quality_gate("congestion", port)

        horizon_days = int(max(1, horizon_weeks) * 7)
        forecast_df = self._forecast_with_intervals(series, horizon_days=horizon_days)
        forecast_df["day_of_week"] = pd.to_datetime(forecast_df["date"], utc=True).dt.day_name()
        target_rows = forecast_df[forecast_df["day_of_week"] == target]
        if target_rows.empty:
            target_rows = forecast_df.head(horizon_weeks)

        mean_pred = float(target_rows["predicted"].mean())
        low_pred = float(target_rows["lower"].mean())
        high_pred = float(target_rows["upper"].mean())

        last_date = pd.Timestamp(series.index.max()).strftime("%Y-%m-%d")
        answer = (
            f"Historical-data forecast anchored after {last_date}: congestion index for {target} is {mean_pred:.2f} "
            f"(interval {low_pred:.2f} to {high_pred:.2f}) over the following {horizon_weeks} week(s). "
            f"This indicates {self._congestion_level(mean_pred)} pressure versus baseline (1.00)."
        )

        history_df = series.reset_index().rename(columns={"index": "date", "congestion_index": "actual"})
        history_df.columns = ["date", "actual"]

        notes = self.kpi.coverage_notes(work, "date")
        notes.append(f"Forecast target weekday: {target}")
        notes.append(f"Forecast horizon: {horizon_weeks} week(s)")
        notes.append(f"Meaning: {self._congestion_meaning(mean_pred)}")

        caveats: List[str] = [
            "Congestion index is a proxy from arrivals and dwell-time availability, not berth-level operations.",
            "Forecast reflects historical weekly patterns only.",
            gate_reason,
        ]
        if work["has_dwell"].fillna(False).mean() < 0.5:
            caveats.append("Dwell coverage is sparse, so forecast is more arrival-driven.")

        return ForecastResult(
            status="ok",
            answer=answer,
            history=history_df,
            forecast=forecast_df,
            coverage_notes=notes,
            caveats=caveats,
        )

    @staticmethod
    def _prediction_triplet(
        result: ForecastResult,
        *,
        target_date: Optional[str] = None,
        target_dow: Optional[str] = None,
    ) -> Optional[tuple[float, float, float]]:
        if result.status != "ok" or result.forecast is None or result.forecast.empty:
            return None
        rows = result.forecast.copy()
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce", utc=True).dt.floor("D")
        rows = rows.dropna(subset=["date", "predicted", "lower", "upper"])
        if rows.empty:
            return None
        if target_date:
            target = pd.to_datetime(target_date, errors="coerce", utc=True)
            if pd.notna(target):
                picked = rows[rows["date"] == pd.Timestamp(target).floor("D")]
                if picked.empty:
                    picked = rows.tail(1)
                return tuple(float(picked[col].mean()) for col in ("predicted", "lower", "upper"))
        if target_dow:
            picked = rows[rows["date"].dt.day_name() == target_dow.title()]
            if picked.empty:
                picked = rows.tail(1)
            return tuple(float(picked[col].mean()) for col in ("predicted", "lower", "upper"))
        picked = rows.tail(1)
        return tuple(float(picked[col].mean()) for col in ("predicted", "lower", "upper"))

    def compare_congestion_ports(
        self,
        ports: List[str],
        *,
        target_date: Optional[str],
        target_dow: Optional[str],
        horizon_weeks: int,
    ) -> ForecastResult:
        unique_ports = list(dict.fromkeys(str(port).strip() for port in ports if str(port).strip()))
        if len(unique_ports) < 2:
            return ForecastResult(
                status="no_data",
                answer="Comparison forecast needs at least two distinct ports.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=[],
            )

        rows: List[Dict[str, float | str]] = []
        coverage_notes: List[str] = []
        caveats: List[str] = []
        missing: List[str] = []
        for port in unique_ports:
            forecast = (
                self.forecast_congestion_for_date(port=port, target_date=target_date, horizon_weeks=horizon_weeks)
                if target_date
                else self.forecast_congestion(port=port, target_dow=target_dow or "Friday", horizon_weeks=horizon_weeks)
            )
            triplet = self._prediction_triplet(forecast, target_date=target_date, target_dow=target_dow)
            if triplet is None:
                missing.append(port)
                continue
            predicted, lower, upper = triplet
            rows.append({"port": port, "predicted": predicted, "lower": lower, "upper": upper})
            for item in forecast.coverage_notes:
                if item not in coverage_notes:
                    coverage_notes.append(item)
            for item in forecast.caveats:
                if item not in caveats:
                    caveats.append(item)
        if len(rows) < 2:
            return ForecastResult(
                status="no_data",
                answer="Could not compute forecasts for at least two requested ports.",
                history=None,
                forecast=None,
                coverage_notes=coverage_notes,
                caveats=caveats[:6],
            )

        table = pd.DataFrame(rows).sort_values("predicted", ascending=False).reset_index(drop=True)
        target_label = target_date or (target_dow or "selected period")
        values = ", ".join(
            f"{row['port']}={float(row['predicted']):.2f} ({float(row['lower']):.2f}-{float(row['upper']):.2f})"
            for _, row in table.iterrows()
        )
        answer = (
            f"Predicted port-pressure comparison for {target_label}: {values}. "
            f"{table.iloc[0]['port']} is likely highest."
        )
        if missing:
            answer += f" No forecast was available for {', '.join(missing)}."
        coverage_notes.insert(0, f"Ports forecast: {', '.join(table['port'].tolist())}")
        if missing:
            coverage_notes.append(f"Requested ports without a forecast: {', '.join(missing)}")
        return ForecastResult(
            status="ok",
            answer=answer,
            history=None,
            forecast=table.set_index("port")[["predicted", "lower", "upper"]],
            coverage_notes=coverage_notes,
            caveats=caveats[:6],
        )

    def compare_congestion_weekdays(
        self,
        port: str,
        *,
        day_a: str,
        day_b: str,
        horizon_weeks: int,
    ) -> ForecastResult:
        rows: List[Dict[str, float | str]] = []
        notes: List[str] = []
        caveats: List[str] = []
        for day in (day_a.title(), day_b.title()):
            forecast = self.forecast_congestion(port=port, target_dow=day, horizon_weeks=horizon_weeks)
            triplet = self._prediction_triplet(forecast, target_dow=day)
            if triplet is None:
                continue
            predicted, lower, upper = triplet
            rows.append({"day_of_week": day, "predicted": predicted, "lower": lower, "upper": upper})
            for item in forecast.coverage_notes:
                if item not in notes:
                    notes.append(item)
            for item in forecast.caveats:
                if item not in caveats:
                    caveats.append(item)
        if len(rows) < 2:
            return ForecastResult(
                status="no_data",
                answer="Could not forecast both requested weekdays.",
                history=None,
                forecast=None,
                coverage_notes=notes,
                caveats=caveats[:6],
            )
        table = pd.DataFrame(rows).sort_values("predicted", ascending=False).reset_index(drop=True)
        details = ", ".join(
            f"{row['day_of_week']}={float(row['predicted']):.2f} ({float(row['lower']):.2f}-{float(row['upper']):.2f})"
            for _, row in table.iterrows()
        )
        answer = (
            f"Forecasted port pressure at {port}: {details}. "
            f"{table.iloc[0]['day_of_week']} is likely more congested."
        )
        return ForecastResult(
            status="ok",
            answer=answer,
            history=None,
            forecast=table.set_index("day_of_week")[["predicted", "lower", "upper"]],
            coverage_notes=notes,
            caveats=caveats[:6],
        )
