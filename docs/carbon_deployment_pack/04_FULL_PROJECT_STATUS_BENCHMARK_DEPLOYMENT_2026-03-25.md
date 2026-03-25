# Eagle Eye Full Project Status, Benchmark Refresh, Conference-Paper Recovery, and Deployment Runbook

Date: 2026-03-25  
Workspace: `/Users/praharshchintu/Documents/New project`

## 1) What is currently working
- Core Streamlit app runtime (local).
- KPI/traffic deterministic analytics.
- Retrieval evidence pipeline with trace/provenance fields.
- Carbon state gating logic (`COMPUTED`, `COMPUTED_ZERO`, `NOT_COMPUTABLE`, `RETRIEVAL_ONLY`, `FORECAST_ONLY`, `UNSUPPORTED`).
- Reliability test suite for intent and carbon-state behavior.

## 2) What is currently failing or weak
- Carbon deterministic coverage is incomplete for many ports/date scopes.
- Result: valid traffic evidence can exist while deterministic carbon output is unavailable.
- Public URL stability depends on your local machine + tunnel process (not an always-on hosted service).
- Conference paper PDF still contains placeholders and duplicated abstract block.

## 3) Refreshed benchmark (latest run)
Source: `/Users/praharshchintu/Documents/New project/evaluation/thesis/results/summary.json`

Top-K: 5  
Queries per strategy: 12

| Strategy | Hit@5 | Mean Latency (ms) | Hallucination Rate | Index Chunks | Index Size (MB) |
|---|---:|---:|---:|---:|---:|
| A (event chunk) | 0.5000 | 158.208 | 0.0000 | 40,000 | 216.351 |
| B (port-day chunk) | 0.4167 | 33.936 | 0.0000 | 7,727 | 239.346 |
| C (hybrid window) | 0.3333 | 25.004 | 0.0000 | 4,054 | 256.418 |

Interpretation:
- Best retrieval quality (Hit@5): Strategy A.
- Best latency: Strategy C.
- Practical trade-off: A for accuracy-sensitive retrieval, C for speed-focused demos.

## 4) Current data coverage snapshot (truth from artifacts)
Source: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/_project_truth_snapshot.json`

### 4.1 Processed row counts
- `events.parquet`: 1,417,791
- `arrivals_daily.parquet`: 103,851
- `arrivals_hourly.parquet`: 355,688
- `dwell_time.parquet`: 82,423
- `congestion_daily.parquet`: 85,605
- `carbon_segments.parquet`: 11,792
- `carbon_emissions_segment.parquet`: 11,792
- `carbon_emissions_daily_port.parquet`: 9,117
- `carbon_emissions_call.parquet`: 87
- `carbon_evidence.parquet`: 11,792

### 4.2 Carbon deterministic availability (critical)
- Carbon segment rows: 11,792
- Call-linked rows: 429
- Call-linked ratio: 3.64%
- Unique `port_key`: 2,218
- Unique `locode_norm`: 2
- Daily rows with non-empty `locode_norm`: 105 / 9,117 (1.15%)

Operational implication:
- Deterministic carbon by port/date often falls back to proxy daily inventory or becomes `NOT_COMPUTABLE`, depending on scope.
- This is a data-coverage limitation, not only a UI issue.

## 5) Why some carbon queries return NOT_COMPUTABLE
The carbon engine intentionally avoids fake numeric output when deterministic scope is missing.

Typical conditions leading to `NOT_COMPUTABLE`:
1. No matching deterministic carbon rows in selected port/date window.
2. Missing call-link support for call-level asks.
3. Carbon artifacts exist but requested identifiers (MMSI/call_id) do not match current table.

This is correct behavior under strict truthfulness rules.  
What needs improvement is carbon coverage and call-link completeness, not forcing synthetic totals.

## 6) Conference paper status audit (from current PDF)
Audited file: `/Users/praharshchintu/Downloads/Eagle_Eye_Conference (5).pdf` (19 pages)

Detected issues:
1. Placeholder affiliations still present (`Affiliation line 1/2/3`).
2. Placeholder corresponding author line still present.
3. Abstract appears duplicated.

Recommended immediate paper fixes:
1. Replace title-page placeholders with final author metadata.
2. Keep only one abstract block.
3. Re-export PDF and re-verify first page text extraction.
4. Ensure benchmark table matches latest numbers above (A/B/C hit@5 + latency).

## 7) Online access options for ChatGPT-agent review

### Option A (fastest, free, temporary, full parity with your local data)
- Run Streamlit locally from this machine.
- Expose with tunnel URL.
- Keep terminal, app process, and tunnel alive.

Pros:
- Uses full local `data/processed` + `data/chroma`.
- Matches local behavior.

Cons:
- URL is session-bound unless you run a named tunnel/domain setup.
- Not always-on.

### Option B (stable hosted app)
- Requires paid or self-managed infra due full data size and vector-store constraints.
- Recommended only when you need always-on availability.

## 8) Exact run commands (known-good sequence)
Run from: `/Users/praharshchintu/Documents/New project`

### 8.1 Clean stop
```bash
pkill -f "streamlit run app/streamlit_app.py" || true
pkill -f "cloudflared tunnel" || true
pkill -f "lt --port 8501" || true
lsof -nP -iTCP:8501 -sTCP:LISTEN || true
```

### 8.2 Local app only (most reliable)
```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
set -a; source .env; set +a
PYTHONPATH=. .venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
Open: `http://127.0.0.1:8501`

### 8.3 Free public URL (if Docker is flaky, use local-process mode in script)
```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
set -a; source .env; set +a
PREFERRED_TUNNEL=cloudflared ./run_free_public_app.sh
```

If you need `eagle-eye.loca.lt` style URL:
```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
set -a; source .env; set +a
nohup PYTHONPATH=. .venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 >/tmp/eagle-eye-streamlit.log 2>&1 &
nohup lt --port 8501 --subdomain eagle-eye >/tmp/eagle-eye-lt.log 2>&1 &
```
Public URL will be `https://eagle-eye.loca.lt` if subdomain is available.

## 9) Troubleshooting (the recurring errors you reported)

### 9.1 `Port 8501 is not available`
```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
pkill -f "streamlit run app/streamlit_app.py" || true
```
Then rerun local app command.

### 9.2 `Docker daemon is not reachable`
- Start Docker Desktop fully and wait until `docker info` works.
- Or bypass Docker and run local Streamlit directly (`8.2`).

### 9.3 `503 Tunnel Unavailable`
- Tunnel process died or URL stale.
- Restart tunnel command only, keep Streamlit alive.

### 9.4 App skeleton keeps loading forever
- Usually tunnel issue or Streamlit process crashed.
- Check `/tmp/eagle-eye-streamlit.log` and tunnel logs.

## 10) What has been completed vs what remains

### Completed
- Carbon state gating to prevent fake zeros and misleading percentages.
- Separation of deterministic carbon evidence vs retrieved supporting traffic evidence.
- Reliability tests for parser/routing/carbon states.
- Benchmark rerun with updated hit@k and latency numbers.
- Carbon/deployment documentation pack.

### Remaining (highest priority)
1. Increase deterministic carbon coverage (especially call-linked rows and locode completeness).
2. Add stable hosted endpoint if always-on public testing is mandatory.
3. Final paper cleanup and regeneration (placeholders + duplicated abstract + benchmark refresh table).

## 11) Strict recommendation for your next demo/paper cycle
1. Demo: run local native Streamlit + tunnel from one clean terminal.
2. Paper: update title page + abstract + benchmark table from latest `summary.json`.
3. Carbon claims: explicitly mark `estimated/proxy-based` and avoid claiming full deterministic coverage for all ports.
4. Include a “data-coverage limitations” subsection in Results/Discussion.

## 12) Source-of-truth files for review
- Benchmarks: `/Users/praharshchintu/Documents/New project/evaluation/thesis/results/summary.json`
- Per-query benchmark details: `/Users/praharshchintu/Documents/New project/evaluation/thesis/results/per_query_results.csv`
- Carbon docs pack index: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/00_INDEX.md`
- Carbon full spec: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/01_CARBON_EMISSIONS_FULL_SPEC.md`
- Carbon QA: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/02_CARBON_EVALUATION_AND_QA.md`
- Deployment plan: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md`
- Project truth snapshot: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/_project_truth_snapshot.json`

## 13) Sample-query reliability validation (latest smoke pass)
Source artifact: `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/_sample_query_smoke_20260325.json`

Validated sample set:
- Traffic Monitoring: 8 queries
- Vessel Investigation: 8 queries
- Forecast Planning: 10 queries
- Carbon & Emissions: 10 queries
- Unsupported Scope: 7 queries
- Total: 43 queries

Observed result summary:
- Exceptions: 0
- Traffic Monitoring: `ok` for all 8
- Vessel Investigation: `ok` for all 8
- Forecast Planning: `ok` for all 10
- Carbon & Emissions: `ok` with `COMPUTED` state for all 10 sample queries
- Unsupported Scope: correctly classified as `unsupported` for all 7

Note:
- This smoke pass proves parser/routing stability for shipped sample queries.
- It does not override data-availability constraints for arbitrary user queries outside covered scopes.
