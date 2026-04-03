from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


HEADING_LINES = [
    "Answer",
    "Answer Source",
    "Carbon Contract",
    "Evidence",
    "Confidence",
    "Chart",
    "How This Was Computed",
    "How To Reduce Emissions",
    "Port Operations Recommendations",
    "Retrieval Provenance",
    "Forecast Meaning",
    "Findings",
    "Emissions Level (Relative Scale)",
]


@dataclass
class CheckResult:
    name: str
    status: str
    message: str = ""


class UIAuditError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _wait_core_ui(page: Any, timeout_ms: int) -> None:
    """Wait for Streamlit to render essential anchors."""
    page.goto("about:blank", wait_until="domcontentloaded", timeout=timeout_ms)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def _safe_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _extract_section(page_text: str, heading: str) -> str:
    lines = _safe_lines(page_text)
    try:
        idx = lines.index(heading)
    except ValueError:
        return ""

    out: List[str] = []
    for j in range(idx + 1, len(lines)):
        line = lines[j]
        if line in HEADING_LINES:
            break
        out.append(line)
    return "\n".join(out).strip()


def _contains_all(text: str, expected: List[str]) -> Tuple[bool, str]:
    missing = [item for item in expected if item not in text]
    if missing:
        return False, f"missing expected strings: {missing}"
    return True, ""


def _contains_any(text: str, expected_any: List[str]) -> Tuple[bool, str]:
    if any(item in text for item in expected_any):
        return True, ""
    return False, f"none of expected_any strings found: {expected_any}"


def _build_summary_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Eagle Eye UI Review Summary")
    lines.append("")
    lines.append(f"Run ID: `{payload.get('run_id', 'unknown')}`")
    lines.append(f"Timestamp (UTC): `{payload.get('timestamp_utc', 'unknown')}`")
    lines.append(f"Git SHA: `{payload.get('git_sha', 'unknown')}`")
    lines.append(f"Base URL: `{payload.get('base_url', 'unknown')}`")
    lines.append(f"API Base URL: `{payload.get('api_base_url', 'n/a')}`")
    lines.append(f"Overall Status: `{payload.get('overall_status', 'unknown')}`")
    lines.append("")

    totals = payload.get("totals", {})
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Scenarios: `{totals.get('scenarios', 0)}`")
    lines.append(f"- Passed: `{totals.get('passed', 0)}`")
    lines.append(f"- Failed: `{totals.get('failed', 0)}`")
    lines.append(f"- API checks passed: `{totals.get('api_passed', 0)}`")
    lines.append(f"- API checks failed: `{totals.get('api_failed', 0)}`")
    lines.append("")

    lines.append("## Scenario Results")
    lines.append("")
    lines.append("| id | category | status | error_code | result_state | screenshot |")
    lines.append("|---|---|---|---|---|---|")
    for sc in payload.get("scenarios", []):
        lines.append(
            "| {id} | {category} | {status} | {error_code} | {result_state} | {shot} |".format(
                id=sc.get("id", ""),
                category=sc.get("category", ""),
                status=sc.get("status", ""),
                error_code=sc.get("error_code", ""),
                result_state=(sc.get("extracted") or {}).get("carbon_result_state", ""),
                shot=", ".join(sc.get("screenshots", [])) or "-",
            )
        )
    lines.append("")

    lines.append("## API Sanity")
    lines.append("")
    lines.append("| name | status | http_code | latency_ms | message |")
    lines.append("|---|---|---:|---:|---|")
    for item in payload.get("api_checks", []):
        lines.append(
            "| {name} | {status} | {code} | {lat} | {msg} |".format(
                name=item.get("name", ""),
                status=item.get("status", ""),
                code=item.get("http_code", ""),
                lat=item.get("latency_ms", ""),
                msg=str(item.get("message", "")).replace("|", "/"),
            )
        )
    lines.append("")

    return "\n".join(lines)


def _write_artifacts(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "review_summary.md").write_text(_build_summary_markdown(payload), encoding="utf-8")


def _scenario_checks(page_text: str, scenario: Dict[str, Any], extracted: Dict[str, Any]) -> List[CheckResult]:
    checks: List[CheckResult] = []
    expected = scenario.get("expected", {})

    required_sections = expected.get("required_sections", [])
    for section in required_sections:
        if _extract_section(page_text, section):
            checks.append(CheckResult(name=f"section:{section}", status="pass", message="present"))
        else:
            checks.append(CheckResult(name=f"section:{section}", status="fail", message="missing"))

    contains_all = expected.get("contains_all", [])
    if contains_all:
        ok, msg = _contains_all(page_text, contains_all)
        checks.append(CheckResult(name="contains_all", status="pass" if ok else "fail", message=msg or "ok"))

    contains_any = expected.get("contains_any", [])
    if contains_any:
        ok, msg = _contains_any(page_text, contains_any)
        checks.append(CheckResult(name="contains_any", status="pass" if ok else "fail", message=msg or "ok"))

    forbidden = expected.get("forbidden_contains", [])
    for token in forbidden:
        if token in page_text:
            checks.append(CheckResult(name=f"forbidden:{token}", status="fail", message="found forbidden token"))
        else:
            checks.append(CheckResult(name=f"forbidden:{token}", status="pass", message="not present"))

    state_any = expected.get("result_state_any", [])
    if state_any:
        state = extracted.get("carbon_result_state", "")
        checks.append(
            CheckResult(
                name="result_state_any",
                status="pass" if state in state_any else "fail",
                message=f"state={state}, expected any of {state_any}",
            )
        )

    return checks


def _classify_error(exc: Exception) -> Tuple[str, str]:
    msg = str(exc)
    lowered = msg.lower()
    if "timeout" in lowered:
        return "render_timeout", msg
    if "ask" in lowered or "question" in lowered:
        return "query_submit_failed", msg
    if "section" in lowered or "missing" in lowered:
        return "section_missing", msg
    return "validation_failed", msg


def _find_question_input(page: Any, timeout_ms: int) -> Any:
    """Locate question input with semantic-first selectors and fallbacks."""
    candidates: List[Any] = []
    try:
        candidates.append(page.get_by_label("Question"))
    except Exception:
        pass
    try:
        candidates.append(page.get_by_placeholder(re.compile(r"question", re.IGNORECASE)))
    except Exception:
        pass
    candidates.append(page.locator("textarea").first)

    for cand in candidates:
        try:
            cand.wait_for(timeout=timeout_ms)
            return cand
        except Exception:
            continue
    raise UIAuditError("section_missing", "Question input not found")


def _find_ask_button(page: Any, timeout_ms: int) -> Any:
    candidates = [
        page.get_by_role("button", name="Ask").first,
        page.get_by_text("Ask", exact=True).first,
        page.locator("button:has-text('Ask')").first,
    ]
    for cand in candidates:
        try:
            cand.wait_for(timeout=timeout_ms)
            return cand
        except Exception:
            continue
    raise UIAuditError("query_submit_failed", "Ask button not found")


def _fill_textbox(page: Any, label: str, value: str, timeout_ms: int) -> None:
    candidates = [
        page.get_by_role("textbox", name=label).first,
        page.locator(f"input[aria-label='{label}']").first,
        page.get_by_label(label).first,
    ]
    for cand in candidates:
        try:
            cand.wait_for(timeout=timeout_ms)
            cand.fill(value)
            return
        except Exception:
            continue
    raise UIAuditError("query_submit_failed", f"Could not fill textbox for label '{label}'")


def _extract_fields(page_text: str) -> Dict[str, Any]:
    answer = _extract_section(page_text, "Answer")
    answer_source = _extract_section(page_text, "Answer Source")
    confidence = _extract_section(page_text, "Confidence")
    prov = _extract_section(page_text, "Retrieval Provenance")

    state_match = re.search(r"Result state:\s*`?([A-Z_]+)`?", page_text)
    status_match = re.search(r"Status:\s*`?([A-Z_]+)`?", page_text)

    sections_present = [h for h in HEADING_LINES if _extract_section(page_text, h)]

    return {
        "answer_preview": answer[:700],
        "answer_source_preview": answer_source[:400],
        "confidence_preview": confidence[:300],
        "retrieval_provenance_preview": prov[:400],
        "carbon_result_state": state_match.group(1) if state_match else "",
        "retrieval_status": status_match.group(1) if status_match else "",
        "sections_present": sections_present,
    }


def _run_api_checks(api_base_url: Optional[str], timeout_s: float = 20.0) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    if not api_base_url:
        return checks

    scenarios = [
        ("health", "GET", "/health", None),
        ("root", "GET", "/", None),
        (
            "ask",
            "POST",
            "/ask",
            {
                "question": "How many vessel arrivals were recorded at SEGOT in March 2022?",
                "top_k_evidence": 3,
                "filters": {"port": "SEGOT", "date_from": "2022-03-01", "date_to": "2022-03-31"},
            },
        ),
        (
            "carbon_ports",
            "GET",
            "/api/v1/carbon/ports/SETRG/emissions?from=2022-03-01&to=2022-03-31&group_by=day&boundary=TTW&pollutants=CO2",
            None,
        ),
    ]

    for name, method, path, payload in scenarios:
        t0 = time.perf_counter()
        url = api_base_url.rstrip("/") + path
        try:
            if method == "GET":
                resp = requests.get(url, timeout=timeout_s)
            else:
                resp = requests.post(url, json=payload, timeout=timeout_s)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            status = "pass" if 200 <= resp.status_code < 300 else "fail"
            msg = "ok" if status == "pass" else (resp.text[:300] if resp.text else "non-2xx")
            checks.append(
                {
                    "name": name,
                    "status": status,
                    "http_code": resp.status_code,
                    "latency_ms": latency_ms,
                    "message": msg,
                }
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            checks.append(
                {
                    "name": name,
                    "status": "fail",
                    "http_code": 0,
                    "latency_ms": latency_ms,
                    "message": str(exc),
                }
            )

    return checks


def run_ui_audit(
    base_url: str,
    scenarios_path: Path,
    output_dir: Path,
    api_base_url: Optional[str] = None,
    headless: bool = True,
    timeout_ms: int = 45_000,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    # Lazy import so non-UI tests don't require Playwright installed.
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError(f"Scenario fixture must be a non-empty list: {scenarios_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    scenario_results: List[Dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1720, "height": 1100})

        for sc in scenarios:
            sc_id = str(sc.get("id", "unknown"))
            category = str(sc.get("category", "uncategorized"))
            query = str(sc.get("query", "")).strip()
            mandatory = bool(sc.get("mandatory", True))
            screenshot_rel = f"screenshots/{sc_id}.png"
            screenshot_abs = output_dir / screenshot_rel
            started = time.perf_counter()

            checks: List[CheckResult] = []
            extracted: Dict[str, Any] = {}
            status = "fail"
            error_code = ""
            error_message = ""
            attempt_errors: List[Dict[str, str]] = []

            for attempt in range(1, max(1, max_attempts) + 1):
                try:
                    page.goto(base_url, wait_until="domcontentloaded", timeout=timeout_ms)
                    main = page.locator("section[data-testid='stMain'], section.main, .block-container").first
                    main.wait_for(timeout=timeout_ms)

                    # Explicit render wait policy for Streamlit.
                    page.wait_for_timeout(1200)
                    page_text = main.inner_text(timeout=timeout_ms)
                    if "Sample Queries" not in page_text or "Ask" not in page_text:
                        raise UIAuditError(
                            "render_timeout",
                            "Core UI anchors (Sample Queries/Ask) not visible after render wait",
                        )

                    question_box = _find_question_input(page=page, timeout_ms=timeout_ms)
                    question_box.click(timeout=timeout_ms)
                    try:
                        question_box.fill("")
                    except Exception:
                        pass
                    question_box.fill(query)

                    filters = sc.get("filters") or {}
                    if any(v not in (None, "", False) for v in filters.values()):
                        try:
                            page.get_by_text("Optional filters", exact=False).first.click(timeout=5000)
                        except Exception:
                            pass
                        if filters.get("port"):
                            try:
                                _fill_textbox(page=page, label="Port / LOCODE / name", value=str(filters["port"]), timeout_ms=timeout_ms)
                            except Exception:
                                _fill_textbox(page=page, label="Port / LOCODE", value=str(filters["port"]), timeout_ms=timeout_ms)
                        if filters.get("date_from"):
                            try:
                                _fill_textbox(page=page, label="From date (YYYY-MM-DD)", value=str(filters["date_from"]), timeout_ms=timeout_ms)
                            except Exception:
                                pass
                        if filters.get("date_to"):
                            try:
                                _fill_textbox(page=page, label="To date (YYYY-MM-DD)", value=str(filters["date_to"]), timeout_ms=timeout_ms)
                            except Exception:
                                pass
                        if filters.get("vessel_type"):
                            _fill_textbox(page=page, label="Vessel type", value=str(filters["vessel_type"]), timeout_ms=timeout_ms)

                    ask_button = _find_ask_button(page=page, timeout_ms=timeout_ms)
                    ask_button.click(timeout=timeout_ms)

                    # Wait for response rendering.
                    page.get_by_text("Answer", exact=True).first.wait_for(timeout=timeout_ms)
                    page.wait_for_timeout(1500)

                    page_text = main.inner_text(timeout=timeout_ms)
                    extracted = _extract_fields(page_text)
                    checks = _scenario_checks(page_text=page_text, scenario=sc, extracted=extracted)

                    if any(c.status == "fail" for c in checks):
                        status = "fail"
                        error_code = "validation_failed"
                        failed = [f"{c.name}: {c.message}" for c in checks if c.status == "fail"]
                        error_message = "; ".join(failed)[:900]
                    else:
                        status = "pass"
                        checks.append(CheckResult(name="scenario", status="pass", message="all checks passed"))
                    break
                except PlaywrightTimeoutError as exc:
                    status = "fail"
                    error_code = "render_timeout"
                    error_message = str(exc)
                    attempt_errors.append({"attempt": str(attempt), "code": error_code, "message": error_message[:500]})
                except UIAuditError as exc:
                    status = "fail"
                    error_code = exc.code
                    error_message = str(exc)
                    attempt_errors.append({"attempt": str(attempt), "code": error_code, "message": error_message[:500]})
                except Exception as exc:
                    status = "fail"
                    error_code, error_message = _classify_error(exc)
                    attempt_errors.append({"attempt": str(attempt), "code": error_code, "message": error_message[:500]})

                # Retry with short cool-down.
                if attempt < max(1, max_attempts):
                    page.wait_for_timeout(1000)

            try:
                page.screenshot(path=str(screenshot_abs), full_page=True)
            except Exception:
                pass

            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            scenario_results.append(
                {
                    "id": sc_id,
                    "category": category,
                    "query": query,
                    "mandatory": mandatory,
                    "status": status,
                    "error_code": error_code,
                    "error_message": error_message,
                    "latency_ms": latency_ms,
                    "attempts": max(1, max_attempts),
                    "attempt_errors": attempt_errors,
                    "checks": [c.__dict__ for c in checks],
                    "extracted": extracted,
                    "screenshots": [screenshot_rel],
                }
            )

        browser.close()

    api_checks = _run_api_checks(api_base_url=api_base_url)

    mandatory_failures = [s for s in scenario_results if s.get("mandatory", True) and s.get("status") != "pass"]
    api_failures = [a for a in api_checks if a.get("status") != "pass"]

    overall_status = "pass" if (not mandatory_failures and not api_failures) else "fail"

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "timestamp_utc": _now_utc(),
        "git_sha": _git_sha(),
        "base_url": base_url,
        "api_base_url": api_base_url,
        "overall_status": overall_status,
        "totals": {
            "scenarios": len(scenario_results),
            "passed": sum(1 for s in scenario_results if s["status"] == "pass"),
            "failed": sum(1 for s in scenario_results if s["status"] != "pass"),
            "api_passed": sum(1 for a in api_checks if a["status"] == "pass"),
            "api_failed": sum(1 for a in api_checks if a["status"] != "pass"),
        },
        "scenarios": scenario_results,
        "api_checks": api_checks,
    }

    _write_artifacts(output_dir=output_dir, payload=payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Eagle Eye UI audit and produce machine-readable review artifacts.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8501", help="Base URL for Streamlit UI")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000", help="Base URL for API sanity checks")
    parser.add_argument("--no-api-checks", action="store_true", help="Skip API sanity checks")
    parser.add_argument("--scenarios", default="review/scenarios.json", help="Path to scenario fixture JSON")
    parser.add_argument("--output-dir", default="review/latest", help="Output directory for review artifacts")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
    parser.add_argument("--timeout-ms", type=int, default=45_000, help="Per-step timeout in milliseconds")
    parser.add_argument("--max-attempts", type=int, default=2, help="Scenario retry attempts on transient UI failures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        raise SystemExit(f"Scenario fixture not found: {scenarios_path}")

    api_base_url = None if args.no_api_checks else args.api_base_url
    payload = run_ui_audit(
        base_url=args.base_url,
        scenarios_path=scenarios_path,
        output_dir=output_dir,
        api_base_url=api_base_url,
        headless=not args.headed,
        timeout_ms=args.timeout_ms,
        max_attempts=args.max_attempts,
    )

    print(json.dumps({
        "overall_status": payload.get("overall_status"),
        "run_id": payload.get("run_id"),
        "totals": payload.get("totals"),
        "output_dir": str(output_dir),
    }, indent=2))

    if payload.get("overall_status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
