"""Backtest forecast quality for arrivals/congestion proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.forecast.forecast import ForecastEngine


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Stabilize percentage errors for near-zero congestion proxy values.
    denom = np.maximum(np.abs(y_true), 1.0)
    values = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(values) * 100.0)


def backtest_metric(
    engine: ForecastEngine,
    metric: str,
    min_history_days: int = 60,
    test_days: int = 28,
    max_ports: int = 20,
) -> Dict[str, object]:
    if metric == "congestion_index":
        base = engine.kpi.congestion.copy()
        value_col = "congestion_index"
        date_col = "date"
    else:
        base = engine.kpi.arrivals_daily.copy()
        value_col = "arrivals_vessels"
        date_col = "date"

    if base.empty:
        return {"metric": metric, "skipped": True, "reason": f"No data for {metric}"}

    port_rank = (
        base.groupby("port_key", dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(max_ports)
        .index.tolist()
    )

    rows: List[Dict[str, object]] = []
    for port in port_rank:
        if metric == "congestion_index":
            port_df = engine.kpi._filter_port(engine.kpi.congestion, port)
            if "pressure_kind" in port_df.columns and (port_df["pressure_kind"] == "full").any():
                port_df = port_df[port_df["pressure_kind"] == "full"]
        else:
            port_df = engine.kpi._filter_port(engine.kpi.arrivals_daily, port)

        series = (
            port_df.groupby(date_col, dropna=False)[value_col]
            .sum()
            .sort_index()
            .astype(float)
        )

        if len(series) < (min_history_days + test_days):
            continue

        train = series.iloc[:-test_days]
        test = series.iloc[-test_days:]
        history = train.tolist()
        preds: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []
        naive_preds: List[float] = []

        for actual in test.tolist():
            pred, lower, upper = engine._one_step_interval(history)
            preds.append(pred)
            lowers.append(lower)
            uppers.append(upper)
            naive_preds.append(float(history[-7] if len(history) >= 7 else history[-1]))
            history.append(float(actual))

        y_true = np.array(test.tolist(), dtype=float)
        y_pred = np.array(preds, dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = _mape(y_true, y_pred)
        naive_mae = float(np.mean(np.abs(y_true - np.array(naive_preds, dtype=float))))
        mase = float(mae / naive_mae) if naive_mae > 0 else float("inf")
        interval_coverage = float(
            np.mean(
                (y_true >= np.array(lowers, dtype=float))
                & (y_true <= np.array(uppers, dtype=float))
            )
        )
        gate_passed = bool(mase < 1.0 and 0.70 <= interval_coverage <= 0.90)

        rows.append(
            {
                "port_key": port,
                "mae": mae,
                "mape": mape,
                "seasonal_naive_mae": naive_mae,
                "mase": mase,
                "interval_80_coverage": interval_coverage,
                "gate_passed": gate_passed,
                "test_points": int(len(test)),
            }
        )

    if not rows:
        return {
            "metric": metric,
            "skipped": True,
            "reason": "No ports with enough history for backtest.",
        }

    df = pd.DataFrame(rows).sort_values("mae")
    finite_mase = pd.to_numeric(df["mase"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mase_mean = float(finite_mase.mean()) if finite_mase.notna().any() else None
    per_port = df.replace([np.inf, -np.inf], np.nan).astype(object)
    per_port = per_port.where(pd.notna(per_port), None)
    return {
        "metric": metric,
        "skipped": False,
        "ports_evaluated": int(len(df)),
        "mae_mean": float(df["mae"].mean()),
        "mape_mean": float(df["mape"].mean()),
        "seasonal_naive_mae_mean": float(df["seasonal_naive_mae"].mean()),
        "mase_mean": mase_mean,
        "interval_80_coverage_mean": float(df["interval_80_coverage"].mean()),
        "ports_passing_gate": int(df["gate_passed"].sum()),
        "per_port": per_port.to_dict(orient="records"),
    }


def run_backtest(
    processed_dir: str | Path = "data/processed",
    out_path: str | Path = "data/processed/forecast_backtest.json",
    min_history_days: int = 60,
    test_days: int = 28,
    max_ports: int = 20,
) -> Dict[str, object]:
    engine = ForecastEngine(processed_dir=processed_dir)
    arrivals_metrics = backtest_metric(
        engine,
        metric="arrivals_vessels",
        min_history_days=min_history_days,
        test_days=test_days,
        max_ports=max_ports,
    )
    congestion_metrics = backtest_metric(
        engine,
        metric="congestion_index",
        min_history_days=min_history_days,
        test_days=test_days,
        max_ports=max_ports,
    )

    payload = {
        "settings": {
            "processed_dir": str(processed_dir),
            "min_history_days": min_history_days,
            "test_days": test_days,
            "max_ports": max_ports,
            "quality_gate": {
                "mase_lt": 1.0,
                "interval_80_coverage_min": 0.70,
                "interval_80_coverage_max": 0.90,
                "maximum_horizon_days": 56,
            },
        },
        "arrivals": arrivals_metrics,
        "congestion": congestion_metrics,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run forecast backtest for arrivals/congestion proxies.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out", default="data/processed/forecast_backtest.json")
    parser.add_argument("--min_history_days", type=int, default=60)
    parser.add_argument("--test_days", type=int, default=28)
    parser.add_argument("--max_ports", type=int, default=20)
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    payload = run_backtest(
        processed_dir=args.processed_dir,
        out_path=args.out,
        min_history_days=args.min_history_days,
        test_days=args.test_days,
        max_ports=args.max_ports,
    )
    arrivals = payload["arrivals"]
    congestion = payload["congestion"]

    if arrivals.get("skipped"):
        print(f"Arrivals backtest skipped: {arrivals.get('reason')}")
    else:
        print(
            "Arrivals backtest:",
            f"ports={arrivals['ports_evaluated']}",
            f"MAE={arrivals['mae_mean']:.3f}",
            f"MAPE={arrivals['mape_mean']:.2f}%",
        )

    if congestion.get("skipped"):
        print(f"Congestion backtest skipped: {congestion.get('reason')}")
    else:
        print(
            "Congestion backtest:",
            f"ports={congestion['ports_evaluated']}",
            f"MAE={congestion['mae_mean']:.3f}",
            f"MAPE={congestion['mape_mean']:.2f}%",
        )

    print(f"Backtest output: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
