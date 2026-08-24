# Eagle Eye runtime, testing, and data guide

## Default product path

Eagle Eye uses the custom React maritime-operations workspace as its default
UI. FastAPI serves the frontend and canonical query service on the same origin.

```bash
cd /Users/praharshchintu/dev/EagleEye
./run_eagle_eye.sh
```

Open `http://127.0.0.1:8000`. Navigation includes Overview, Analysis Desk,
Traffic Monitoring, Vessel Investigation, ETA Watch, Port Pressure, and the
exact visible `Carbon Emissions` label. `Analysis Desk` retains the internal
`Chat Assistant` identifier for compatibility; ETA Watch retains `/eta-delay`
and the internal `ETA & Delay` identifier. Advanced and voyage-grade modes are
not in public navigation.

The unchanged Streamlit QA and rollback surface remains available on port
`8501`:

```bash
PORT=8501 ./run_streamlit.sh
```

## Data boundary

- Deterministic analytics use only versioned structured datasets.
- Historical coverage currently ends on 2022-04-30 and is never presented as
  live operational truth.
- ETA Watch is a separately bounded AISStream workflow. It uses fresh
  vessel-reported destination, ETA, position, speed, and ETA-revision
  observations for a curated Sweden-first Baltic catalog. It is non-exhaustive,
  not an official port arrival board, and not a confirmed-delay or prediction
  source. Queries are limited to the next 48 hours.
- Other current traffic or weather requests return an explicit unavailable
  state.
- Research checks the local document collection before authoritative sources.
- General and research model routes are opt-in with
  `EAGLE_EYE_ENABLE_MODEL_RESPONSES=true`; do not enable them until the API key
  has been rotated.
- Query endpoints are read-only. Files are created only by
  `POST /api/v2/exports`.
- New public responses publish verified High assurance or a specific
  unavailable/not-applicable state. Medium and Low are retained only for
  displaying legacy stored responses, never upgraded cosmetically.
- A live ETA source outage, stale AIS report, ambiguous vessel, unsupported
  port, or out-of-horizon request returns a machine-readable unavailable state
  and never falls back to historical congestion.

The active schema, table hashes, row counts, coverage, ports, enabled
operations, and model-validation results are recorded in
`data/processed/data_manifest.json`.

## Public interfaces

- `POST /api/v2/query`
- `POST /api/v2/query/stream`
- `GET /api/v2/capabilities`
- `POST /api/v2/exports`
- `POST /api/v2/feedback`

`POST /ask` and `POST /api/v1/chat` are compatibility adapters over the same
canonical query service.

## Verification

```bash
./scripts/verify_all.sh
```

The release verifier runs secret scanning, the full Python suite and frozen
human-authored query regressions, canonical/compatibility API checks, the web
build, dependency audit, Playwright visual and responsive tests, axe
accessibility checks, and an offline FastAPI/served-UI smoke test. It disables
model responses and removes `OPENAI_API_KEY` from the verification process.

The separate canonical response inspector remains in
`src/app/streamlit_canonical.py` for backend QA and renders the same
`AnswerEnvelope` as FastAPI.
