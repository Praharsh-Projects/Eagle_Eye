#!/usr/bin/env python3
"""Build the legacy v3 compatibility catalog.

This classifier-derived artifact is retained only for historical regression
compatibility. The release-blocking oracle is the frozen, human-authored
``evaluation/gold/query_regressions_v4.json`` suite.
"""

from __future__ import annotations

import ast
import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.kpi.query import KPIQueryEngine
from src.qa.intent import classify_question


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "evaluation/latest/prompt_contract_v3.json"

PAPER_CASES = [
    "How many vessel arrivals were recorded at Gothenburg in March 2022?",
    "Total vessel arrivals at Gothenburg in March 2022?",
    "How many vessel arrivals were recorded at Karlshamn and Gothenburg in March 2022?",
    "According to port-call records, show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28.",
    "According to port-call records, how many tanker arrivals were recorded at LVVNT between 2022-03-01 and 2022-03-10?",
    "What is crane utilization at berth 3 in SEGOT today?",
]


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt).strip()).casefold()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _load_ui_prompts() -> Dict[str, List[str]]:
    source_path = ROOT / "src/app/streamlit_app.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "SAMPLE_QUERIES_BY_CATEGORY":
                value = ast.literal_eval(node.value)
                return {str(key): [str(item) for item in items] for key, items in value.items()}
    raise RuntimeError("SAMPLE_QUERIES_BY_CATEGORY was not found in Streamlit source.")


def _add_case(
    cases: "OrderedDict[str, Dict[str, Any]]",
    prompt: str,
    source_tag: str,
    source_expectation: str | None = None,
) -> None:
    normalized = _normalize_prompt(prompt)
    item = cases.setdefault(
        normalized,
        {
            "prompt": re.sub(r"\s+", " ", str(prompt).strip()),
            "source_tags": [],
            "source_expectations": [],
        },
    )
    if source_tag not in item["source_tags"]:
        item["source_tags"].append(source_tag)
    if source_expectation and source_expectation not in item["source_expectations"]:
        item["source_expectations"].append(source_expectation)


def _load_source_cases() -> "OrderedDict[str, Dict[str, Any]]":
    cases: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for category, prompts in _load_ui_prompts().items():
        for prompt in prompts:
            _add_case(cases, prompt, f"ui:{_slug(category)}")

    catalog = json.loads((ROOT / "evaluation/latest/query_catalog_v2.json").read_text(encoding="utf-8"))
    for category, items in catalog["categories"].items():
        for item in items:
            _add_case(
                cases,
                item["query"],
                f"catalog:{_slug(category)}",
                str(item.get("expected_result_state") or ""),
            )

    legacy = json.loads(
        (ROOT / "docs/carbon_deployment_pack/_sample_query_smoke_20260325.json").read_text(encoding="utf-8")
    )
    for item in legacy["results"]:
        legacy_state = str(item.get("result_state") or item.get("status") or "")
        _add_case(cases, item["query"], f"legacy:{_slug(item['category'])}", legacy_state)

    for prompt in PAPER_CASES:
        _add_case(cases, prompt, "paper:critical")

    if len(cases) != 75:
        raise RuntimeError(f"Expected 75 normalized prompts, found {len(cases)}.")
    return cases


def _canonical_ports(tokens: Iterable[str], engine: KPIQueryEngine) -> List[str]:
    resolved: List[str] = []
    for token in tokens:
        value = engine.resolve_port_token(str(token)) or str(token).strip()
        if value and value not in resolved:
            resolved.append(value)
    return resolved


def _arrival_oracle(
    rows: pd.DataFrame,
    locode: str,
    start: str,
    end: str,
    vessel_type: str | None = None,
    allow_day_gap_fallback: bool = False,
) -> tuple[int, int]:
    work = rows[
        (rows["locode_norm"].fillna("").astype(str).str.upper() == locode)
        | (rows["port_key"].fillna("").astype(str).str.upper() == locode)
    ].copy()
    work = work[work["date"].between(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))]
    if vessel_type:
        work = work[
            (work["source_kind"] == "port_call")
            & (work["vessel_type_norm"].fillna("").astype(str).str.casefold() == vessel_type.casefold())
        ]
    else:
        structured = work[work["source_kind"] == "port_call"]
        if allow_day_gap_fallback:
            structured_days = set(structured["date"].dt.floor("D").tolist())
            proxy = work[
                (work["source_kind"] != "port_call")
                & ~work["date"].dt.floor("D").isin(structured_days)
            ]
            work = pd.concat([structured, proxy], ignore_index=True)
        else:
            work = structured
    # Historical arrivals are port-call events.  The daily distinct-vessel
    # field is retained for pressure/forecast features and must not be used as
    # an arrival-count oracle.
    return int(work["arrivals_events"].sum()), int(work["date"].dt.floor("D").nunique())


def _paper_oracles() -> Dict[str, List[Dict[str, Any]]]:
    rows = pd.read_parquet(ROOT / "data/processed/arrivals_daily.parquet")
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce", utc=True)

    segot, segot_days = _arrival_oracle(rows, "SEGOT", "2022-03-01", "2022-03-31")
    sekan, _ = _arrival_oracle(
        rows,
        "SEKAN",
        "2022-03-01",
        "2022-03-31",
        allow_day_gap_fallback=True,
    )
    lvvnt, lvvnt_days = _arrival_oracle(rows, "LVVNT", "2022-02-01", "2022-02-28")
    tankers, tanker_days = _arrival_oracle(
        rows,
        "LVVNT",
        "2022-03-01",
        "2022-03-10",
        vessel_type="tanker",
    )
    derived = {
        "segot": segot,
        "segot_days": segot_days,
        "multi": segot + sekan,
        "lvvnt": lvvnt,
        "lvvnt_days": lvvnt_days,
        "tankers": tankers,
        "tanker_days": tanker_days,
    }
    required = {
        "segot": 488,
        "segot_days": 31,
        "multi": 563,
        "lvvnt": 58,
        "lvvnt_days": 25,
        "tankers": 7,
        "tanker_days": 5,
    }
    if derived != required:
        raise RuntimeError(f"Structured paper oracles drifted: expected {required}, derived {derived}.")

    values: Dict[str, List[Dict[str, Any]]] = {}
    segot_prompts = [
        "How many vessel arrivals were recorded at SEGOT in March 2022?",
        "How many vessel arrivals were recorded at Gothenburg in March 2022?",
        "Total vessel arrivals at Gothenburg in March 2022?",
    ]
    for prompt in segot_prompts:
        values[_normalize_prompt(prompt)] = [
            {"name": "arrival_count", "value": segot, "tolerance": 0},
            {"name": "day_buckets", "value": segot_days, "tolerance": 0},
        ]
    values[_normalize_prompt(PAPER_CASES[2])] = [
        {"name": "combined_arrival_count", "value": segot + sekan, "tolerance": 0}
    ]
    values[_normalize_prompt(PAPER_CASES[3])] = [
        {"name": "arrival_count", "value": lvvnt, "tolerance": 0},
        {"name": "day_buckets", "value": lvvnt_days, "tolerance": 0},
    ]
    values[_normalize_prompt(PAPER_CASES[4])] = [
        {"name": "tanker_arrivals", "value": tankers, "tolerance": 0},
        {"name": "day_buckets", "value": tanker_days, "tolerance": 0},
    ]
    return values


def _expected_states(prompt: str, intent: str) -> List[str]:
    q = prompt.casefold()
    if intent == "G":
        return ["UNSUPPORTED"]
    if intent == "C":
        return ["FORECAST_ONLY"]
    if intent == "H":
        if "unknownport" in q or "2035-" in q or "no deterministic rows" in q:
            return ["NOT_COMPUTABLE", "RETRIEVAL_ONLY"]
        if "predict carbon" in q:
            return ["FORECAST_ONLY"]
        return ["COMPUTED", "COMPUTED_ZERO"]
    if "route travel time" in q and "compare arrivals" not in q:
        return ["NOT_COMPUTABLE"]
    return ["COMPUTED"]


def _coverage_mode(prompt: str, states: List[str]) -> str:
    q = prompt.casefold()
    if states == ["UNSUPPORTED"]:
        return "unsupported"
    if "karlshamn and karlskrona" in q or ("compare arrivals" in q and "route durations" in q):
        return "partial"
    if "NOT_COMPUTABLE" in states and "COMPUTED" not in states:
        return "unavailable"
    if "FORECAST_ONLY" in states:
        return "forecast"
    return "complete"


def _required_terms(prompt: str, intent: str, coverage: str, boundary: str | None) -> List[str]:
    q = prompt.casefold()
    terms: List[str] = []
    if coverage == "partial":
        terms.append("partial coverage")
    if "resolve voyage" in q or "segment timeline" in q or "voyage evidence" in q:
        terms.append("retired Voyage Lab")
    elif intent == "G":
        terms.append("outside AIS/port-call scope")
    if boundary == "TTW_WTW":
        terms.extend(["TTW", "WTW"])
    if "evidence id" in q and "no deterministic rows" not in q:
        terms.append("Evidence IDs")
    if "predict carbon" in q:
        terms.append("forecast outputs are not available")
    if "according to port-call records" in q:
        terms.append("port-call")
    return terms


def build_contract() -> Dict[str, Any]:
    source_cases = _load_source_cases()
    kpi = KPIQueryEngine(processed_dir=ROOT / "data/processed")
    required_values = _paper_oracles()
    contract_cases: List[Dict[str, Any]] = []

    for index, (normalized, source_case) in enumerate(source_cases.items(), start=1):
        prompt = source_case["prompt"]
        classified = classify_question(prompt)
        entities = classified.entities
        canonical_ports = _canonical_ports(entities.get("ports") or [], kpi)
        relative_date = bool(re.search(r"\b(?:next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", prompt, re.I))
        states = _expected_states(prompt, classified.intent)
        coverage = _coverage_mode(prompt, states)
        boundary = str(entities.get("boundary")) if classified.intent == "H" else None
        forbidden_claims: List[str] = []
        forbid_numeric_answer = classified.intent == "G"
        if classified.intent == "G":
            forbidden_claims = [
                "utilization is",
                "queue length is",
                "productivity is",
                "occupancy is",
                "resolved voyage id",
            ]

        contract_cases.append(
            {
                "case_id": f"pcv3_{index:03d}",
                "prompt": prompt,
                "normalized_prompt": normalized,
                "source_tags": sorted(source_case["source_tags"]),
                "source_expectations": sorted(source_case["source_expectations"]),
                "expectations": {
                    "intents": [classified.intent],
                    "metric": str(entities.get("metric") or ""),
                    "entities": {
                        "ports": canonical_ports,
                        "mmsi": entities.get("mmsi"),
                        "call_id": entities.get("call_id"),
                        "vessel_type": entities.get("vessel_type"),
                        "dow": entities.get("dow"),
                        "dow_compare": entities.get("dow_compare"),
                        "aggregation": entities.get("aggregation"),
                    },
                    "dates": {
                        "mode": "relative" if relative_date else "absolute",
                        "date_from": None if relative_date else entities.get("date_from"),
                        "date_to": None if relative_date else entities.get("date_to"),
                        "target_date": None if relative_date else entities.get("target_date"),
                    },
                    "result_states": states,
                    "coverage": coverage,
                    "boundary": boundary,
                    "required_values": required_values.get(normalized, []),
                    "required_terms": _required_terms(prompt, classified.intent, coverage, boundary),
                    "forbidden_claims": forbidden_claims,
                    "forbid_numeric_answer": forbid_numeric_answer,
                },
            }
        )

    return {
        "schema_version": "3.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(contract_cases),
        "normalization": "Unicode casefold plus collapsed whitespace",
        "numeric_oracle": "Derived independently from data/processed/arrivals_daily.parquet; legacy answers are provenance only.",
        "cases": contract_cases,
    }


def main() -> int:
    contract = build_contract()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {contract['case_count']} cases to {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
