# Eagle Eye Validation Checklist (Strict Pass/Fail)

Date: 2026-03-25  
Workspace: `/Users/praharshchintu/Documents/New project`

## A) Code/Test Integrity

| Check | Status | Evidence |
|---|---|---|
| Carbon+intent reliability test suite executes | PASS | `python -m unittest tests.test_intent_reliability tests.test_carbon_query_states tests.test_carbon_presentation` -> `Ran 20 tests ... OK` |
| Benchmark rerun completes | PASS | `python -m src.thesis.evaluate ...` completed and rewrote summary artifacts |
| Summary benchmark file updated | PASS | `/Users/praharshchintu/Documents/New project/evaluation/thesis/results/summary.json` |

## B) Query Coverage Smoke (in-app sample set)

Source: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/_sample_query_smoke_20260325.json`

| Category | Queries | Result | Status |
|---|---:|---|---|
| Traffic Monitoring | 8 | all `ok` | PASS |
| Vessel Investigation | 8 | all `ok` | PASS |
| Forecast Planning | 10 | all `ok` | PASS |
| Carbon & Emissions | 10 | all `ok` with `COMPUTED` state | PASS |
| Unsupported Scope | 7 | all `unsupported` | PASS |
| Exceptions | 43 total | 0 exceptions | PASS |

## C) Runtime Reachability

| Check | Status | Notes |
|---|---|---|
| Local Streamlit health (`127.0.0.1:8501/_stcore/health`) | PASS | Verified `200` in latest local run |
| `eagle-eye.loca.lt` stable free subdomain | FAIL (intermittent) | Returned `503`/`000` in this validation cycle |
| Random free tunnel URL reliability | PARTIAL | Works in some sessions; free tunnel endpoints are not guaranteed stable |

## D) Carbon Data Availability (deterministic scope quality)

| Check | Status | Evidence |
|---|---|---|
| Deterministic carbon tables present | PASS | `carbon_emissions_*`, `carbon_segments`, `carbon_evidence` exist in `data/processed` |
| Deterministic call-level coverage broad enough for all ports | FAIL | `carbon_emissions_call.parquet` has 87 rows only |
| Port-normalized locode coverage sufficient | FAIL | non-empty `locode_norm` in daily carbon rows ~1.15% |

## E) Conference Paper Readiness (current PDF)

Audited file: `/Users/praharshchintu/Downloads/Eagle_Eye_Conference (5).pdf`

| Check | Status | Notes |
|---|---|---|
| Placeholder affiliations removed | FAIL | `Affiliation line ...` placeholders still present |
| Corresponding author finalized | FAIL | placeholder still present |
| Single abstract only | FAIL | duplicate abstract detected |
| Benchmark values aligned with latest run | PARTIAL | must be manually synced to refreshed `summary.json` |

## F) Conclusion
- Core system logic and sample-query routing are stable.
- Main blockers are deployment stability on free public tunnels and limited deterministic carbon coverage for arbitrary port/date scopes.
- Conference paper requires immediate metadata and abstract cleanup before submission.
