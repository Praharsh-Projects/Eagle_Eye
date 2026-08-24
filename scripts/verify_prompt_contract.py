#!/usr/bin/env python3
"""Execute the legacy classifier-derived v3 compatibility catalog.

This is not a release oracle. ``scripts/verify_all.sh`` runs the frozen,
human-authored v4 query regressions through the canonical service.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.app.streamlit_app import _handle_ask_question
from src.carbon.query import CarbonQueryEngine, CarbonResult
from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.qa.intent import classify_question


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "evaluation/latest/prompt_contract_v3.json"
RESULTS_PATH = ROOT / "evaluation/latest/prompt_contract_v3_results.json"
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")


def _result_state(result: AnalyticsResult | ForecastResult | CarbonResult) -> str:
    if isinstance(result, CarbonResult):
        return str(result.result_state)
    if isinstance(result, ForecastResult):
        return "FORECAST_ONLY" if result.status == "ok" else "NOT_COMPUTABLE"
    if result.status == "ok":
        return "COMPUTED"
    if result.status == "unsupported":
        return "UNSUPPORTED"
    return "NOT_COMPUTABLE"


def _combined_text(result: AnalyticsResult | ForecastResult | CarbonResult) -> str:
    values: List[str] = [str(result.answer or "")]
    values.extend(str(item) for item in (result.coverage_notes or []))
    values.extend(str(item) for item in (result.caveats or []))
    if isinstance(result, CarbonResult):
        values.extend([str(result.source_label or ""), str(result.confidence_reason or "")])
    return "\n".join(values)


def _number_values(text: str) -> List[float]:
    values: List[float] = []
    for match in NUMBER_RE.findall(text or ""):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def _result_numbers(result: AnalyticsResult | ForecastResult | CarbonResult) -> List[float]:
    values = _number_values(str(result.answer or ""))
    table = getattr(result, "table", None)
    if isinstance(table, pd.DataFrame) and not table.empty:
        for column in table.select_dtypes(include="number").columns:
            values.extend(pd.to_numeric(table[column], errors="coerce").dropna().astype(float).tolist())
    return values


def _contains_value(values: Iterable[float], target: float, tolerance: float) -> bool:
    return any(math.isclose(float(item), float(target), abs_tol=float(tolerance), rel_tol=0.0) for item in values)


def _canonical_ports(tokens: Iterable[str], engine: KPIQueryEngine) -> List[str]:
    result: List[str] = []
    for token in tokens:
        resolved = engine.resolve_port_token(str(token)) or str(token).strip()
        if resolved and resolved not in result:
            result.append(resolved)
    return result


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[rank])


def _validate_case(
    case: Dict[str, Any],
    kpi: KPIQueryEngine,
    forecast: ForecastEngine,
    carbon: CarbonQueryEngine,
    events_path: Path,
) -> Dict[str, Any]:
    prompt = str(case["prompt"])
    expected = case["expectations"]
    classified = classify_question(prompt)
    errors: List[str] = []

    if classified.intent not in expected["intents"]:
        errors.append(f"intent {classified.intent!r} not in {expected['intents']!r}")
    if str(classified.entities.get("metric") or "") != str(expected["metric"]):
        errors.append(
            f"metric {classified.entities.get('metric')!r} != {expected['metric']!r}"
        )

    expected_dates = expected["dates"]
    if expected_dates["mode"] == "absolute":
        for key in ("date_from", "date_to", "target_date"):
            if classified.entities.get(key) != expected_dates.get(key):
                errors.append(
                    f"{key} {classified.entities.get(key)!r} != {expected_dates.get(key)!r}"
                )

    started = time.perf_counter()
    result, _ = _handle_ask_question(
        question=prompt,
        intent_result=classified,
        kpi=kpi,
        forecaster=forecast,
        carbon=carbon,
        retriever=None,
        top_k_evidence=0,
        user_filters={},
        events_path=events_path if events_path.exists() else None,
    )
    elapsed = time.perf_counter() - started
    actual_state = _result_state(result)
    if actual_state not in expected["result_states"]:
        errors.append(f"result state {actual_state!r} not in {expected['result_states']!r}")

    actual_ports = _canonical_ports(classified.entities.get("ports") or [], kpi)
    expected_ports = expected["entities"]["ports"]
    if actual_ports != expected_ports:
        errors.append(f"ports {actual_ports!r} != {expected_ports!r}")
    for key in ("mmsi", "call_id", "vessel_type", "dow", "dow_compare", "aggregation"):
        if classified.entities.get(key) != expected["entities"].get(key):
            errors.append(
                f"entity {key} {classified.entities.get(key)!r} != {expected['entities'].get(key)!r}"
            )

    if expected["boundary"] is not None:
        actual_boundary = getattr(result, "boundary", classified.entities.get("boundary"))
        if str(actual_boundary) != str(expected["boundary"]):
            errors.append(f"boundary {actual_boundary!r} != {expected['boundary']!r}")

    combined_text = _combined_text(result)
    lowered = combined_text.casefold()
    for term in expected["required_terms"]:
        if str(term).casefold() not in lowered:
            errors.append(f"required term missing: {term!r}")
    for claim in expected["forbidden_claims"]:
        if str(claim).casefold() in lowered:
            errors.append(f"forbidden claim present: {claim!r}")

    if expected["coverage"] == "partial" and "partial coverage" not in lowered:
        errors.append("partial-coverage answer is not labeled explicitly")
    if expected["coverage"] == "unsupported" and actual_state != "UNSUPPORTED":
        errors.append("unsupported coverage did not produce UNSUPPORTED")

    values = _result_numbers(result)
    for requirement in expected["required_values"]:
        target = float(requirement["value"])
        tolerance = float(requirement.get("tolerance", 0.0))
        if not _contains_value(values, target, tolerance):
            errors.append(
                f"required value {requirement['name']}={target:g} was not present"
            )

    if expected["forbid_numeric_answer"]:
        prompt_numbers = _number_values(prompt)
        unexpected = [
            value
            for value in _number_values(str(result.answer or ""))
            if not _contains_value(prompt_numbers, value, 0.0)
        ]
        if unexpected:
            errors.append(f"unsupported answer fabricated numeric values: {unexpected!r}")

    return {
        "case_id": case["case_id"],
        "prompt": prompt,
        "source_tags": case["source_tags"],
        "intent": classified.intent,
        "metric": classified.entities.get("metric"),
        "result_state": actual_state,
        "elapsed_seconds": round(elapsed, 4),
        "answer": str(result.answer or ""),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    if not CONTRACT_PATH.exists():
        raise SystemExit(f"Missing {CONTRACT_PATH}. Run scripts/build_prompt_contract.py first.")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("case_count") != 75 or len(contract.get("cases") or []) != 75:
        raise SystemExit("Prompt contract must contain exactly 75 cases.")

    processed_dir = ROOT / "data/processed"
    kpi = KPIQueryEngine(processed_dir=processed_dir)
    forecast = ForecastEngine(processed_dir=processed_dir)
    carbon = CarbonQueryEngine(
        processed_dir=processed_dir,
        factor_registry_path=ROOT / "config/carbon_factors.v1.json",
        monte_carlo_draws=100,
        auto_build=False,
    )
    events_path = processed_dir / "events.parquet"

    results = [
        _validate_case(case, kpi, forecast, carbon, events_path)
        for case in contract["cases"]
    ]
    durations = [float(item["elapsed_seconds"]) for item in results]
    failed = [item for item in results if not item["passed"]]
    p50 = statistics.median(durations) if durations else 0.0
    p95 = _percentile(durations, 0.95)
    if p95 > 2.0:
        failed.append(
            {
                "case_id": "latency_gate",
                "prompt": "deterministic p95",
                "errors": [f"p95 {p95:.3f}s exceeds 2.000s"],
            }
        )

    report = {
        "schema_version": "3.0",
        "case_count": len(results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "failed_count": sum(1 for item in results if not item["passed"]),
        "latency_seconds": {
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "max": round(max(durations) if durations else 0.0, 4),
        },
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(
        f"Prompt contract: {report['passed_count']}/{report['case_count']} passed; "
        f"p50={p50:.3f}s p95={p95:.3f}s"
    )
    for item in failed:
        print(f"FAIL {item['case_id']}: {item['prompt']}")
        for error in item["errors"]:
            print(f"  - {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
