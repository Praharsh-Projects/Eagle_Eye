from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.app.streamlit_app import _handle_ask_question
from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.qa.intent import classify_question
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation" / "competitive_compare" / "latest"
CHART_DIR = OUT_DIR / "charts"
TABLE_DIR = OUT_DIR / "tables"


def _silence_streamlit_logging() -> None:
    for name in [
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.state.session_state_proxy",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


@dataclass
class PromptArtifact:
    question: str
    intent: str
    aggregation: str | None
    status: str
    answer: str
    table_rows: int
    coverage_notes: list[str]
    resolved_scope: dict[str, Any]


@dataclass
class ExactParityResult:
    task_id: str
    vendor_surface: str
    question: str
    expected_total: int | None
    eagle_eye_total: int | None
    series_exact_match: bool | None
    baseline_preview: list[dict[str, Any]]
    eagle_eye_answer: str
    eagle_eye_status: str
    parity_status: str
    notes: str


@dataclass
class EagleEyeOnlyResult:
    task_id: str
    question: str | None
    questions: list[str] | None
    output_summary: str
    evidence: str
    why_not_one_step_in_templates: str


@dataclass
class CaveatResult:
    task_id: str
    question: str
    raw_event_total: int
    eagle_eye_total: int
    parity_status: str
    explanation: str


EXACT_PARITY_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "lvvnt_feb_2022_daily_arrivals",
        "vendor_surface": "MarineTraffic Port Calls / VesselFinder Port Calls API",
        "question": "According to port-call records, show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28.",
        "port": "LVVNT",
        "start": "2022-02-01",
        "end": "2022-02-28",
        "vessel_type": None,
        "kind": "arrivals",
        "notes": "Exact parity on both daily counts and monthly total when Eagle Eye is explicitly scoped to port-call records.",
    },
    {
        "task_id": "lvvnt_tanker_2022_03_01_10",
        "vendor_surface": "MarineTraffic Port Calls filter / VesselFinder Port Calls API filter",
        "question": "According to port-call records, how many tanker arrivals were recorded at LVVNT between 2022-03-01 and 2022-03-10?",
        "port": "LVVNT",
        "start": "2022-03-01",
        "end": "2022-03-10",
        "vessel_type": "tanker",
        "kind": "arrivals",
        "notes": "Replicates a filtered tanker-arrival count that existing DSS tools can expose through port-call filters or APIs.",
    },
    {
        "task_id": "plgdn_march_2022_arrivals",
        "vendor_surface": "MarineTraffic Port Details statistics / VesselFinder Port Calls API",
        "question": "According to port-call records, how many vessel arrivals were recorded at PLGDN in March 2022?",
        "port": "PLGDN",
        "start": "2022-03-01",
        "end": "2022-03-31",
        "vessel_type": None,
        "kind": "arrivals",
        "notes": "Shows parity on a higher-volume port window rather than only a small-sample case.",
    },
    {
        "task_id": "sekan_march_2022_arrivals",
        "vendor_surface": "MarineTraffic Port Calls / VesselFinder Port Calls API",
        "question": "According to port-call records, how many vessel arrivals were recorded at SEKAN in March 2022?",
        "port": "SEKAN",
        "start": "2022-03-01",
        "end": "2022-03-31",
        "vessel_type": None,
        "kind": "arrivals",
        "notes": "Useful small-port business case: exact parity on a monthly arrivals query for a southern Baltic port.",
    },
    {
        "task_id": "karlshamn_first_arrival_2022_03_22",
        "vendor_surface": "MarineTraffic Port Calls table sorted by arrival / VesselFinder Port Calls API sorted by arrival",
        "question": "According to port-call records, the first arrival seen at Karlshamn on 2022-03-22",
        "port": "Karlshamn",
        "start": "2022-03-22",
        "end": "2022-03-22",
        "vessel_type": None,
        "kind": "first_arrival",
        "notes": "Exact parity on earliest-arrival identification from the same port-call dataset.",
    },
]

EAGLE_EYE_ONLY_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "multi_port_aggregate",
        "question": "According to port-call records, how many vessel arrivals were recorded at Karlshamn and Karlskrona in March 2022?",
        "why": "MarineTraffic and VesselFinder expose port-level tables and APIs, but this cross-port total is not a one-step end-user query in their fixed UI surfaces. It normally requires two filtered exports and external aggregation.",
    },
    {
        "task_id": "equivalent_phrasing_consistency",
        "questions": [
            "How many vessel arrivals were recorded at SEGOT in March 2022?",
            "How many arrivals were recorded at SEGOT in March 2022?",
            "Total vessel arrivals at Gothenburg in March 2022?",
            "Count vessel arrivals at the Port of Gothenburg in March 2022.",
        ],
        "why": "The end-user can phrase the same analytical request in different vocabulary and still receive the same resolved result. Template DSS products do not expose this language-normalization capability as the primary interaction mode.",
    },
    {
        "task_id": "first_route_vessel",
        "question": "The first vessel from Szczecin to Swinoujscie in March 2021",
        "why": "This requires reconstructed voyage episodes and route-level sorting. Existing DSS tools may expose logs, but not typically as a direct analyst-facing natural-language query.",
    },
    {
        "task_id": "route_travel_time_summary",
        "question": "What is the median and p90 route travel time from Szczecin to Swinoujscie between 2021-02-01 and 2021-03-31?",
        "why": "This is a derived route statistic over reconstructed voyages. Existing tracking tools usually require export and external processing to compute route medians and percentiles.",
    },
    {
        "task_id": "unsupported_scope_refusal",
        "question": "What is crane utilization at berth 3 in SEGOT today?",
        "why": "Eagle Eye explicitly refuses terminal-internal metrics outside the dataset scope. Fixed tracking tools do not usually express this as a natural-language refusal contract because they are not query-first systems.",
    },
]

SEMANTIC_CAVEAT_TASK = {
    "task_id": "segot_port_call_semantic_gap",
    "question": "According to port-call records, how many vessel arrivals were recorded at SEGOT in March 2022?",
    "port": "SEGOT",
    "start": "2022-03-01",
    "end": "2022-03-31",
}

CAPABILITY_ROWS: list[dict[str, str]] = [
    {
        "tool": "MarineTraffic",
        "class": "Commercial maritime DSS",
        "official_url": "https://support.marinetraffic.com/en/articles/9552786-port-details-page",
        "documented_surface": "Port details, port calls, expected arrivals, congestion/statistics views",
        "natural_language_front_door": "No public analyst prompt surface",
        "one_step_multi_port_aggregate": "No public one-step query surface; requires repeated filters/export",
        "route_percentile_summary": "Not publicly documented as one-step analyst query",
        "answer_contract": "View/filter/export oriented, not answer-contract oriented",
        "paper_positioning": "Strong parity comparator for fixed-template port-call and statistics workflows",
    },
    {
        "tool": "VesselFinder",
        "class": "Commercial maritime DSS / API",
        "official_url": "https://www.vesselfinder.com/port-calls-api",
        "documented_surface": "Port Calls API, historical AIS, voyage, vessel data",
        "natural_language_front_door": "No public analyst prompt surface",
        "one_step_multi_port_aggregate": "Possible via API retrieval, but requires external aggregation",
        "route_percentile_summary": "Requires external post-processing",
        "answer_contract": "API payload oriented, not answer-contract oriented",
        "paper_positioning": "Good comparator for structured maritime data access without ad hoc NL analyst orchestration",
    },
    {
        "tool": "Windward MAI Expert",
        "class": "Maritime AI / intelligence assistant",
        "official_url": "https://windward.ai/solutions/mai-expert/",
        "documented_surface": "AI-assisted maritime intelligence and investigation workflow",
        "natural_language_front_door": "Yes, product is explicitly AI-driven",
        "one_step_multi_port_aggregate": "Not publicly evidenced from available docs",
        "route_percentile_summary": "Not publicly evidenced from available docs",
        "answer_contract": "AI/intelligence oriented; explicit deterministic KPI contract not publicly documented",
        "paper_positioning": "AI comparator for maritime intelligence positioning rather than raw port-call parity",
    },
    {
        "tool": "Kpler Maritime",
        "class": "Maritime intelligence platform",
        "official_url": "https://www.kpler.com/product/maritime",
        "documented_surface": "Maritime data, tracking, analytics, market intelligence",
        "natural_language_front_door": "Not publicly evidenced from product page",
        "one_step_multi_port_aggregate": "Not publicly evidenced",
        "route_percentile_summary": "Not publicly evidenced",
        "answer_contract": "Analytics platform, not publicly documented as deterministic analyst answer contract",
        "paper_positioning": "Market intelligence comparator, not fair raw parity benchmark without product access",
    },
    {
        "tool": "project44 Ocean Visibility",
        "class": "Ocean visibility / ETA platform",
        "official_url": "https://www.project44.com/platform/visibility/ocean/",
        "documented_surface": "Ocean tracking, predictive ETA, exception visibility",
        "natural_language_front_door": "Not publicly evidenced",
        "one_step_multi_port_aggregate": "Not publicly evidenced",
        "route_percentile_summary": "Not publicly evidenced",
        "answer_contract": "Visibility and ETA oriented",
        "paper_positioning": "Useful ETA/visibility comparator, not a direct port-call prompt benchmark",
    },
    {
        "tool": "FourKites Ocean Freight Visibility",
        "class": "Ocean visibility / ETA platform",
        "official_url": "https://www.fourkites.com/platform/ocean-freight-visibility/",
        "documented_surface": "Ocean shipment visibility, ETA, exception management",
        "natural_language_front_door": "Not publicly evidenced",
        "one_step_multi_port_aggregate": "Not publicly evidenced",
        "route_percentile_summary": "Not publicly evidenced",
        "answer_contract": "Visibility oriented",
        "paper_positioning": "Comparator for ETA and visibility category only",
    },
    {
        "tool": "Portcast",
        "class": "Predictive visibility / ETA platform",
        "official_url": "https://www.portcast.io/predictive-visibility",
        "documented_surface": "Predictive ocean visibility, ETA, exception analytics",
        "natural_language_front_door": "Not publicly evidenced",
        "one_step_multi_port_aggregate": "Not publicly evidenced",
        "route_percentile_summary": "Not publicly evidenced",
        "answer_contract": "Predictive visibility oriented",
        "paper_positioning": "Comparator for ETA/predictive visibility, not direct ad hoc port-call querying",
    },
    {
        "tool": "Spire Maritime",
        "class": "AIS data and analytics provider",
        "official_url": "https://spire.com/industry/maritime/",
        "documented_surface": "AIS data feeds, vessel tracking, maritime analytics inputs",
        "natural_language_front_door": "No public analyst prompt surface",
        "one_step_multi_port_aggregate": "Requires downstream analytics implementation",
        "route_percentile_summary": "Requires downstream analytics implementation",
        "answer_contract": "Data-provider oriented",
        "paper_positioning": "Input-data comparator rather than end-user DSS benchmark",
    },
    {
        "tool": "Lloyd's List Intelligence",
        "class": "Maritime intelligence platform",
        "official_url": "https://www.lloydslistintelligence.com/",
        "documented_surface": "Maritime intelligence, fleet and vessel insight",
        "natural_language_front_door": "Not publicly evidenced",
        "one_step_multi_port_aggregate": "Not publicly evidenced",
        "route_percentile_summary": "Not publicly evidenced",
        "answer_contract": "Intelligence platform oriented",
        "paper_positioning": "Broader intelligence comparator; direct workflow parity needs product access",
    },
]


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    rows = [cols] + [["" if pd.isna(v) else str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]

    def fmt(row: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    out = [fmt(rows[0]), sep]
    for row in rows[1:]:
        out.append(fmt(row))
    return "\n".join(out)


class ComparisonRunner:
    def __init__(self) -> None:
        _silence_streamlit_logging()
        config = load_config(str(ROOT / "config/config.yaml"))
        processed_rel = config["paths"].get("processed_dir", "data/processed")
        self.processed_dir = (ROOT / processed_rel).resolve()
        carbon_cfg = config.get("carbon", {})
        self.kpi = KPIQueryEngine(processed_dir=self.processed_dir)
        self.forecast = ForecastEngine(processed_dir=self.processed_dir)
        self.carbon = CarbonQueryEngine(
            processed_dir=self.processed_dir,
            factor_registry_path=str((ROOT / carbon_cfg.get("factor_registry_path", "config/carbon_factors.v1.json")).resolve()),
            monte_carlo_draws=int(carbon_cfg.get("monte_carlo_draws", 500)),
            auto_build=True,
        )
        self.events_path = self.processed_dir / "events.parquet"
        self.raw_port_calls = pd.read_csv(ROOT / "data/PRJ896.csv", low_memory=False)
        self.raw_port_calls["portArrival"] = pd.to_datetime(self.raw_port_calls["portArrival"], errors="coerce", utc=True)
        self.raw_port_calls["portDeparture"] = pd.to_datetime(self.raw_port_calls["portDeparture"], errors="coerce", utc=True)
        self.raw_port_calls["portLocode"] = self.raw_port_calls["portLocode"].fillna("").astype(str).str.upper()
        self.raw_port_calls["portNameNorm"] = self.raw_port_calls["portName"].fillna("").astype(str).str.strip().str.lower()
        self.raw_port_calls["vesselTypeNorm"] = self.raw_port_calls["vesselType"].fillna("").astype(str).str.strip().str.lower()
        self.raw_ais = pd.read_csv(ROOT / "data/PRJ912.csv", low_memory=False)
        self.raw_ais["TimePosition"] = pd.to_datetime(self.raw_ais["TimePosition"], errors="coerce", utc=True)
        self.raw_ais["TimeETA"] = pd.to_datetime(self.raw_ais["TimeETA"], errors="coerce", utc=True)
        self.dwell = pd.read_parquet(self.processed_dir / "dwell_time.parquet")
        self.dwell["arrival_time"] = pd.to_datetime(self.dwell["arrival_time"], errors="coerce", utc=True)
        self.dwell["departure_time"] = pd.to_datetime(self.dwell["departure_time"], errors="coerce", utc=True)
        self.dwell["arrival_date"] = pd.to_datetime(self.dwell["arrival_date"], errors="coerce", utc=True)
        self.port_lookup = self._build_port_lookup()

    def _build_port_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        distinct = self.raw_port_calls[["portLocode", "portName"]].dropna().drop_duplicates()
        for _, row in distinct.iterrows():
            code = str(row["portLocode"]).strip().upper()
            name = str(row["portName"]).strip().lower()
            if code:
                lookup[code] = code
            if name:
                lookup[name] = code
        return lookup

    def _resolve_port(self, token: str) -> str:
        raw = str(token).strip()
        if not raw:
            return raw
        key = raw.lower()
        compact = raw.upper().replace(" ", "")
        return self.port_lookup.get(key) or self.port_lookup.get(compact) or compact

    def run_prompt(self, question: str) -> tuple[PromptArtifact, Any]:
        intent = classify_question(question)
        result, _ = _handle_ask_question(
            question,
            intent,
            self.kpi,
            self.forecast,
            self.carbon,
            None,
            5,
            {},
            self.events_path,
        )
        resolved_scope = dict((intent.entities or {}).get("extraction_diagnostics", {}).get("resolved_scope", {}))
        artifact = PromptArtifact(
            question=question,
            intent=intent.intent,
            aggregation=intent.entities.get("aggregation"),
            status=getattr(result, "status", "unknown"),
            answer=result.answer,
            table_rows=0 if getattr(result, "table", None) is None else int(len(result.table)),
            coverage_notes=list(getattr(result, "coverage_notes", []) or []),
            resolved_scope=resolved_scope,
        )
        return artifact, result

    def _filter_raw_port_calls(
        self,
        port: str,
        start: str,
        end: str,
        vessel_type: str | None = None,
    ) -> pd.DataFrame:
        port_code = self._resolve_port(port)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        if len(end) == 10:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        work = self.raw_port_calls[self.raw_port_calls["portLocode"] == port_code].copy()
        work = work[(work["portArrival"] >= start_ts) & (work["portArrival"] <= end_ts)]
        if vessel_type:
            vt = str(vessel_type).strip().lower()
            work = work[work["vesselTypeNorm"] == vt]
        return work.sort_values("portArrival")

    def _daily_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["date", "count"])
        out = (
            df.assign(date=df["portArrival"].dt.floor("D"))
            .groupby("date", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("date")
            .reset_index(drop=True)
        )
        return out

    def _evaluate_arrivals_task(self, task: dict[str, Any]) -> ExactParityResult:
        artifact, result = self.run_prompt(task["question"])
        baseline = self._filter_raw_port_calls(task["port"], task["start"], task["end"], task.get("vessel_type"))
        baseline_daily = self._daily_counts(baseline)
        eagle_table = result.table.copy() if getattr(result, "table", None) is not None else pd.DataFrame()
        if not eagle_table.empty and "date" in eagle_table.columns:
            eagle_daily = eagle_table[["date", "arrivals_vessels"]].copy()
            eagle_daily["date"] = pd.to_datetime(eagle_daily["date"], errors="coerce", utc=True).dt.floor("D")
            eagle_daily = eagle_daily.rename(columns={"arrivals_vessels": "count"}).sort_values("date").reset_index(drop=True)
            baseline_cmp = baseline_daily.copy()
            baseline_cmp["date"] = pd.to_datetime(baseline_cmp["date"], errors="coerce", utc=True).dt.floor("D")
            series_exact = eagle_daily.equals(baseline_cmp)
            eagle_total = int(eagle_daily["count"].sum())
        else:
            eagle_daily = pd.DataFrame(columns=["date", "count"])
            series_exact = False
            eagle_total = None
        expected_total = int(len(baseline))
        parity_status = "exact_match" if series_exact and eagle_total == expected_total and artifact.status == "ok" else "mismatch"
        preview = baseline_daily.head(10).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
        return ExactParityResult(
            task_id=task["task_id"],
            vendor_surface=task["vendor_surface"],
            question=task["question"],
            expected_total=expected_total,
            eagle_eye_total=eagle_total,
            series_exact_match=bool(series_exact),
            baseline_preview=preview,
            eagle_eye_answer=artifact.answer,
            eagle_eye_status=artifact.status,
            parity_status=parity_status,
            notes=task["notes"],
        )

    def _evaluate_first_arrival_task(self, task: dict[str, Any]) -> ExactParityResult:
        artifact, result = self.run_prompt(task["question"])
        baseline = self._filter_raw_port_calls(task["port"], task["start"], task["end"], task.get("vessel_type"))
        if baseline.empty:
            return ExactParityResult(
                task_id=task["task_id"], vendor_surface=task["vendor_surface"], question=task["question"],
                expected_total=None, eagle_eye_total=None, series_exact_match=None, baseline_preview=[],
                eagle_eye_answer=artifact.answer, eagle_eye_status=artifact.status, parity_status="no_baseline",
                notes="No baseline rows found in raw port-call data.",
            )
        first = baseline.iloc[0]
        baseline_ts = pd.to_datetime(first["portArrival"], errors="coerce", utc=True)
        baseline_mmsi = str(first["vesselMMSI"])
        eagle_match = False
        if getattr(result, "table", None) is not None and not result.table.empty:
            row = result.table.iloc[0]
            eagle_ts = pd.to_datetime(row.get("arrival_time"), errors="coerce", utc=True)
            eagle_mmsi = str(row.get("mmsi"))
            eagle_match = (baseline_mmsi == eagle_mmsi) and (baseline_ts == eagle_ts)
        preview = [{
            "baseline_first_mmsi": baseline_mmsi,
            "baseline_first_arrival_utc": baseline_ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(baseline_ts) else None,
        }]
        return ExactParityResult(
            task_id=task["task_id"],
            vendor_surface=task["vendor_surface"],
            question=task["question"],
            expected_total=None,
            eagle_eye_total=None,
            series_exact_match=eagle_match,
            baseline_preview=preview,
            eagle_eye_answer=artifact.answer,
            eagle_eye_status=artifact.status,
            parity_status="exact_match" if eagle_match and artifact.status == "ok" else "mismatch",
            notes=task["notes"],
        )

    def evaluate_exact_parity(self) -> list[ExactParityResult]:
        out: list[ExactParityResult] = []
        for task in EXACT_PARITY_TASKS:
            if task["kind"] == "arrivals":
                out.append(self._evaluate_arrivals_task(task))
            elif task["kind"] == "first_arrival":
                out.append(self._evaluate_first_arrival_task(task))
        return out

    def evaluate_eagle_eye_only(self) -> list[EagleEyeOnlyResult]:
        results: list[EagleEyeOnlyResult] = []
        for task in EAGLE_EYE_ONLY_TASKS:
            if task.get("questions"):
                prompt_runs = [self.run_prompt(q)[0] for q in task["questions"]]
                answers = {run.answer for run in prompt_runs}
                consistent = len(answers) == 1
                summary = next(iter(answers)) if answers else "No answer"
                evidence = (
                    f"{len(prompt_runs)}/{len(prompt_runs)} prompts resolved to the same answer." if consistent else
                    f"Answers diverged across prompts: {sorted(answers)}"
                )
                results.append(
                    EagleEyeOnlyResult(
                        task_id=task["task_id"],
                        question=None,
                        questions=task["questions"],
                        output_summary=summary,
                        evidence=evidence,
                        why_not_one_step_in_templates=task["why"],
                    )
                )
            else:
                artifact, _ = self.run_prompt(task["question"])
                results.append(
                    EagleEyeOnlyResult(
                        task_id=task["task_id"],
                        question=task["question"],
                        questions=None,
                        output_summary=artifact.answer,
                        evidence=f"status={artifact.status}; intent={artifact.intent}; aggregation={artifact.aggregation}",
                        why_not_one_step_in_templates=task["why"],
                    )
                )
        return results

    def evaluate_semantic_caveat(self) -> CaveatResult:
        artifact, result = self.run_prompt(SEMANTIC_CAVEAT_TASK["question"])
        baseline = self._filter_raw_port_calls(
            SEMANTIC_CAVEAT_TASK["port"],
            SEMANTIC_CAVEAT_TASK["start"],
            SEMANTIC_CAVEAT_TASK["end"],
        )
        raw_total = int(len(baseline))
        eagle_total = None
        if getattr(result, "table", None) is not None and not result.table.empty and "arrivals_vessels" in result.table.columns:
            eagle_total = int(pd.to_numeric(result.table["arrivals_vessels"], errors="coerce").fillna(0).sum())
        parity_status = "semantic_difference" if eagle_total != raw_total else "exact_match"
        explanation = (
            "Raw export totals count port-call events. Eagle Eye source-scoped descriptive traffic answers currently summarize the daily arrivals_vessels metric. "
            "Where the same vessel records multiple call events within the month, totals can diverge even though both outputs are grounded in the same underlying data source."
        )
        return CaveatResult(
            task_id=SEMANTIC_CAVEAT_TASK["task_id"],
            question=SEMANTIC_CAVEAT_TASK["question"],
            raw_event_total=raw_total,
            eagle_eye_total=int(eagle_total or 0),
            parity_status=parity_status,
            explanation=explanation,
        )

    def _plot_daily_parity_overlay(self, port: str, start: str, end: str, out_name: str) -> None:
        raw = self._filter_raw_port_calls(port, start, end)
        raw_daily = self._daily_counts(raw)
        artifact, result = self.run_prompt(
            f"According to port-call records, show daily arrival counts at {port} between {start} and {end}."
        )
        ee = result.table[["date", "arrivals_vessels"]].copy()
        ee["date"] = pd.to_datetime(ee["date"], errors="coerce", utc=True).dt.floor("D")
        ee = ee.rename(columns={"arrivals_vessels": "count"})

        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(raw_daily["date"], raw_daily["count"], marker="o", linewidth=2.0, label="Raw port-call export")
        ax.plot(ee["date"], ee["count"], marker="s", linewidth=1.8, linestyle="--", label="Eagle Eye (source-scoped)")
        ax.set_title(f"Daily arrivals parity: {port} ({start} to {end})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Arrival count")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(CHART_DIR / out_name, dpi=180)
        plt.close(fig)

    def _plot_marine_traffic_style_daily(self, port: str, start: str, end: str, out_name: str) -> None:
        raw = self._filter_raw_port_calls(port, start, end)
        daily = self._daily_counts(raw)
        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.bar(daily["date"], daily["count"], color="#1976d2", width=0.8)
        ax.set_title(f"MarineTraffic-style arrivals by day: {port} ({start} to {end})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Arrivals")
        ax.grid(axis="y", alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(CHART_DIR / out_name, dpi=180)
        plt.close(fig)

    def _plot_marine_traffic_style_vessel_mix(self, port: str, start: str, end: str, out_name: str) -> None:
        raw = self._filter_raw_port_calls(port, start, end)
        mix = raw.groupby("vesselType").size().reset_index(name="count").sort_values("count", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.bar(mix["vesselType"], mix["count"], color="#0288d1")
        ax.set_title(f"MarineTraffic-style arrival mix by vessel type: {port} ({start} to {end})")
        ax.set_xlabel("Vessel type")
        ax.set_ylabel("Arrivals")
        ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(CHART_DIR / out_name, dpi=180)
        plt.close(fig)

    def _plot_port_congestion_style_dwell(self, port: str, start: str, end: str, out_name: str) -> None:
        port_code = self._resolve_port(port)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        work = self.dwell[(self.dwell["port_key"].astype(str).str.upper() == port_code) | (self.dwell["locode_norm"].astype(str).str.upper() == port_code)].copy()
        work = work[(work["arrival_time"] >= start_ts) & (work["arrival_time"] <= end_ts)]
        work = work.dropna(subset=["arrival_time", "dwell_minutes"]).copy()
        work["arrival_day"] = work["arrival_time"].dt.floor("D")
        summary = work.groupby("arrival_day", dropna=False)["dwell_minutes"].median().div(60.0).reset_index(name="median_dwell_h")
        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.plot(summary["arrival_day"], summary["median_dwell_h"], color="#ef6c00", marker="o", linewidth=2)
        ax.set_title(f"Port-congestion-style median time in port: {port} ({start} to {end})")
        ax.set_xlabel("Arrival day")
        ax.set_ylabel("Median dwell (hours)")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(CHART_DIR / out_name, dpi=180)
        plt.close(fig)

    def build_vendor_style_artifacts(self) -> dict[str, str]:
        # Export-style tables.
        plgdn_calls = self._filter_raw_port_calls("PLGDN", "2022-03-01", "2022-03-31")
        port_calls_export = plgdn_calls[[
            "vesselName", "portArrival", "portDeparture", "portName", "portLocode",
            "vesselDestinationArrival", "vesselDestinationDeparture", "vesselType", "vesselMMSI", "vesselIMO",
        ]].rename(columns={
            "vesselName": "vessel_name",
            "portArrival": "arrival_utc",
            "portDeparture": "departure_utc",
            "portName": "port_name",
            "portLocode": "port_locode",
            "vesselDestinationArrival": "voyage_origin_port",
            "vesselDestinationDeparture": "voyage_destination_port",
            "vesselType": "vessel_type",
            "vesselMMSI": "mmsi",
            "vesselIMO": "imo",
        })
        port_calls_export.to_csv(TABLE_DIR / "marinetraffic_style_port_calls_plgdn_2022_03.csv", index=False)

        vessel_master = (
            self.raw_ais.sort_values("TimePosition")
            .drop_duplicates(subset=["MMSI"], keep="last")[[
                "MMSI", "IMO", "Name", "Callsign", "Flag", "VesselType", "Length", "Width", "Draught", "Destination", "TimeETA"
            ]]
            .rename(columns={
                "MMSI": "mmsi",
                "IMO": "imo",
                "Name": "name",
                "Callsign": "callsign",
                "Flag": "flag",
                "VesselType": "vessel_type",
                "Length": "length_m",
                "Width": "width_m",
                "Draught": "draught_m",
                "Destination": "destination",
                "TimeETA": "eta_utc",
            })
            .head(200)
        )
        vessel_master.to_csv(TABLE_DIR / "vesselfinder_style_vessel_master_sample.csv", index=False)

        self._plot_daily_parity_overlay("LVVNT", "2022-02-01", "2022-02-28", "lvvnt_feb2022_daily_parity.png")
        self._plot_marine_traffic_style_daily("PLGDN", "2022-03-01", "2022-03-31", "plgdn_march2022_arrivals_daily.png")
        self._plot_marine_traffic_style_vessel_mix("PLGDN", "2022-03-01", "2022-03-31", "plgdn_march2022_arrivals_by_vessel_type.png")
        self._plot_port_congestion_style_dwell("PLGDN", "2022-03-01", "2022-03-31", "plgdn_march2022_median_dwell.png")

        return {
            "port_calls_export": str(TABLE_DIR / "marinetraffic_style_port_calls_plgdn_2022_03.csv"),
            "vessel_master_export": str(TABLE_DIR / "vesselfinder_style_vessel_master_sample.csv"),
            "parity_chart": str(CHART_DIR / "lvvnt_feb2022_daily_parity.png"),
            "daily_arrivals_chart": str(CHART_DIR / "plgdn_march2022_arrivals_daily.png"),
            "vessel_mix_chart": str(CHART_DIR / "plgdn_march2022_arrivals_by_vessel_type.png"),
            "dwell_chart": str(CHART_DIR / "plgdn_march2022_median_dwell.png"),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_report(
    exact: list[ExactParityResult],
    eagle_only: list[EagleEyeOnlyResult],
    caveat: CaveatResult,
    artifacts: dict[str, str],
    capability_df: pd.DataFrame,
) -> None:
    exact_df = pd.DataFrame([asdict(item) for item in exact])
    eagle_df = pd.DataFrame([asdict(item) for item in eagle_only])
    caveat_df = pd.DataFrame([asdict(caveat)])

    report = f"""# Eagle Eye Competitive Comparison Pack

## Scope
This pack compares Eagle Eye against MarineTraffic/VesselFinder-style workflows on the **same historical port-call data family**, then separates tasks that Eagle Eye can answer in one prompt but fixed-template DSS tools typically cannot answer in a one-step analyst workflow.

## 1. Exact parity tasks (same source family, same output)
{_df_to_markdown(exact_df[['task_id','vendor_surface','expected_total','eagle_eye_total','series_exact_match','parity_status']])}

## 2. Eagle Eye-only tasks
{_df_to_markdown(eagle_df[['task_id','output_summary','evidence']])}

## 3. Semantic caveat
{_df_to_markdown(caveat_df[['task_id','raw_event_total','eagle_eye_total','parity_status']])}

{caveat.explanation}

## 4. Vendor-style local artifacts
- MarineTraffic-style port calls export: `{artifacts['port_calls_export']}`
- VesselFinder-style vessel master sample: `{artifacts['vessel_master_export']}`
- Raw-vs-Eagle Eye parity chart: `{artifacts['parity_chart']}`
- Daily arrivals chart: `{artifacts['daily_arrivals_chart']}`
- Vessel-type mix chart: `{artifacts['vessel_mix_chart']}`
- Port-congestion-style dwell chart: `{artifacts['dwell_chart']}`

## 5. Broader AI/DSS capability map
{_df_to_markdown(capability_df[['tool','class','documented_surface','natural_language_front_door','one_step_multi_port_aggregate','paper_positioning']])}

## 6. Paper-ready interpretation
1. Use the exact-parity tasks to show that Eagle Eye reproduces MarineTraffic/VesselFinder-style outputs when the prompt is explicitly scoped to port-call records.
2. Use the Eagle Eye-only tasks to show the added value of natural-language orchestration over deterministic analytics: multi-port aggregation, phrasing consistency, route-first queries, route percentiles, and explicit refusal.
3. Mention the semantic caveat directly: Eagle Eye's default descriptive KPI path may summarize **unique vessel arrivals** rather than raw **port-call events** unless the prompt or metric is explicitly scoped. This is a modeling choice, not fabricated output.
"""
    (OUT_DIR / "comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    runner = ComparisonRunner()
    exact = runner.evaluate_exact_parity()
    eagle_only = runner.evaluate_eagle_eye_only()
    caveat = runner.evaluate_semantic_caveat()
    artifacts = runner.build_vendor_style_artifacts()
    capability_df = pd.DataFrame(CAPABILITY_ROWS)

    exact_df = pd.DataFrame([asdict(item) for item in exact])
    eagle_df = pd.DataFrame([asdict(item) for item in eagle_only])
    caveat_df = pd.DataFrame([asdict(caveat)])

    exact_df.to_csv(OUT_DIR / "exact_parity_tasks.csv", index=False)
    eagle_df.to_csv(OUT_DIR / "eagle_eye_only_tasks.csv", index=False)
    caveat_df.to_csv(OUT_DIR / "semantic_caveats.csv", index=False)
    capability_df.to_csv(OUT_DIR / "capability_matrix.csv", index=False)

    summary = {
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exact_parity_total": int(len(exact)),
        "exact_parity_matches": int(sum(1 for item in exact if item.parity_status == "exact_match")),
        "eagle_eye_only_total": int(len(eagle_only)),
        "semantic_caveat": asdict(caveat),
        "artifacts": artifacts,
    }
    _write_json(OUT_DIR / "summary.json", summary)
    _write_json(OUT_DIR / "exact_parity_tasks.json", [asdict(item) for item in exact])
    _write_json(OUT_DIR / "eagle_eye_only_tasks.json", [asdict(item) for item in eagle_only])
    _write_json(OUT_DIR / "semantic_caveats.json", [asdict(caveat)])

    _write_report(exact, eagle_only, caveat, artifacts, capability_df)

    print(f"Wrote comparison pack to {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
