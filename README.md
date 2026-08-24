# Eagle Eye Maritime Intelligence

Eagle Eye is a correctness-first maritime analytics workspace. Its public
interface is a custom React maritime-operations workspace served by the
canonical FastAPI service. The previous Streamlit interface remains available
as an internal QA and rollback surface.

The product deliberately separates three modes:

- deterministic analytics over versioned historical data;
- source-grounded maritime research, local documents first;
- general assistance, with current web grounding when explicitly enabled.

Unsupported, ambiguous, or current-operational requests return specific states
instead of falling back to an unrelated metric.

Public results follow a high-or-unavailable assurance policy. A numeric or
current factual claim is published only after its result-specific checks pass;
otherwise the response explains why the result is unavailable. Legacy stored
Medium or Low results are labelled as unverified and are never rerun silently.

The bounded live-data exception is ETA Watch. It uses AISStream vessel
broadcasts for a curated Sweden-first Baltic destination catalog and can report
fresh vessel-reported destinations, ETAs, point positions, speeds, signal
quality, and ETA revisions. The feed is non-exhaustive and is never described
as an official schedule, confirmed delay, arrival confirmation, or Eagle Eye
prediction. The page never substitutes the historical 2022 pressure proxy as a
live ETA.

## Start locally

Prerequisites are Python 3.12, `uv`, and the local data bundle.

```bash
cd /Users/praharshchintu/dev/EagleEye
./scripts/bootstrap_runtime.sh
./run_eagle_eye.sh
```

Open `http://127.0.0.1:8000`.

The React workspace provides Overview, Analysis Desk, Traffic Monitoring,
Vessel Investigation, ETA Watch, Port Pressure, and the exact
`Carbon Emissions` label. Advanced and voyage-grade modes remain absent from
public navigation. `/eta-delay` and the internal `ETA & Delay` identifier remain
unchanged for compatibility.

The production frontend must be built before first launch:

```bash
npm --prefix web ci
npm --prefix web run build
```

To run the unchanged Streamlit QA surface on its traditional port:

```bash
PORT=8501 ./run_streamlit.sh
```

The small canonical response inspector remains available directly in
`src/app/streamlit_canonical.py` for backend QA.

## Security and model routes

Copy `.env.example` to `.env`, keep `.env` at mode `600`, and never commit it.
Model-backed routes are disabled by default and require both a rotated API key
and `EAGLE_EYE_ENABLE_MODEL_RESPONSES=true`. Deterministic analytics remain the
sole numeric authority even when models are enabled.

ETA Watch requires a backend-only `AISSTREAM_API_KEY`. Keep the key in the
mode-`600` local `.env`; it is sent only in the AISStream WebSocket subscription
frame and is never returned by an API response or written to logs.

## API

- `POST /api/v2/query` returns the complete `AnswerEnvelope`.
- `POST /api/v2/query/stream` emits progress and text events, then that same
  final envelope.
- `GET /api/v2/capabilities` reports supported operations and data freshness.
- `POST /api/v2/exports` is the only query-result export path.
- `POST /api/v2/feedback` records a trace-linked issue report for review.

Legacy `/ask` and `/api/v1/chat` routes are adapters over the canonical service.

## Verify

The public source publication can be checked without the restricted thesis
workspace:

```bash
shasum -a 256 -c release/github_source_snapshot.sha256
python3 scripts/secret_scan.py \
  --root . \
  --allowlist release/public_secret_scan_allowlist.json
npm --prefix web ci
npm --prefix web run build
```

The complete local release gate remains `./scripts/verify_all.sh`. It expects
the authorised data and thesis evidence workspace, and runs the complete Python
and frozen regression suites, API hash consistency, production web build,
dependency audit, Playwright responsive/visual tests, axe accessibility checks,
and an offline served-application smoke test. No live model call is made.

See `docs/EAGLE_EYE_PAGES_TESTING_AND_DATA_GUIDE.md` for the data boundary,
runtime details, and endpoint inventory.
