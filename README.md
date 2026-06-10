# Eagle Eye Maritime Data and Decision-Support Platform

Eagle Eye is a portfolio platform for maritime operational analytics. It combines AIS and port-call processing, deterministic KPI queries, congestion forecasting, emissions estimates, carbon evidence, and optional retrieval-grounded answers.

The project is designed around a product principle: numeric answers come from deterministic data services, while RAG is used only for supporting evidence and explanatory context.

## Product Overview

Eagle Eye helps an analyst ask operational questions such as:
- how many vessels arrived at a port during a period
- which day or hour is usually busiest
- whether a port shows congestion or arrival spikes
- what future congestion may look like from historical patterns
- what tank-to-wake or well-to-wake emissions estimates are available
- why the system can or cannot answer a question with the available data

The system exposes both:
- a Streamlit review interface in `src/app/streamlit_app.py`
- a FastAPI service in `src/api/server.py`

## Users and Use Cases

| User | Need | Eagle Eye capability |
| --- | --- | --- |
| Product or operations lead | Understand which data-backed services are feasible | Capability and coverage notes from processed datasets |
| Data/API engineer | Expose consistent operational queries | FastAPI `/ask` plus versioned carbon endpoints |
| Analyst or researcher | Inspect evidence behind an answer | Method steps, provenance, confidence labels, and chart payloads |
| Sustainability stakeholder | Review emissions estimates and limitations | TTW/WTW outputs, uncertainty summaries, and evidence records |

## Implemented Service Capabilities

Deterministic analytics:
- arrival counts and busiest day/hour
- dwell and port-stay summaries where port-call data supports it
- congestion proxy based on arrival and dwell patterns
- arrival spikes and weekday comparisons

Forecasting:
- historical-pattern forecasts for arrivals and congestion
- forecast backtest entry point through `src/forecast/backtest.py`

Carbon and emissions:
- AIS/port-call based segment building
- TTW pollutants: `CO2e`, `NOx`, `SOx`, `PM`
- WTW `CO2e`
- uncertainty intervals, confidence labels, parameter versioning, and evidence tables

Retrieval-grounded evidence:
- optional document and traffic retrieval through Chroma/OpenAI embeddings
- provenance fields including source metadata and retrieved evidence lines
- clean fallback behavior when retrieval or processed data is unavailable

## API Surface

Run the FastAPI service with:

```bash
./run_api.sh
```

Core endpoints:
- `GET /health`
- `POST /ask`
- `GET /api/v1/carbon/ports/{port_id}/emissions`
- `GET /api/v1/carbon/vessels/{mmsi}/calls/{call_id}`
- `POST /api/v1/carbon/estimate`
- `GET /api/v1/carbon/evidence/{evidence_id}`

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What will congestion be at LVVNT on Friday, February 20, 2026?",
    "top_k_evidence": 5,
    "filters": {"port": "LVVNT"}
  }'
```

Swagger docs are available at:

```text
http://localhost:8000/docs
```

## Architecture

```text
raw CSV / optional PDFs / public URLs
  -> src.predict.data_prep
  -> src.kpi.build_kpis
  -> src.carbon.build
  -> optional src.index.build_index
  -> Streamlit UI and FastAPI API
```

Important modules:
- `src/kpi`: deterministic operational KPI build and query layer
- `src/forecast`: historical-pattern forecasting and backtesting
- `src/carbon`: emissions build, query, presentation, and evidence handling
- `src/rag`: retrieval, routing, and evidence formatting
- `src/api/server.py`: API state, `/ask`, and carbon endpoints
- `src/app/streamlit_app.py`: analyst-facing review UI

## Data Inputs

Expected local inputs:
- `data/PRJ912.csv` for AIS telemetry
- `data/PRJ896.csv` for port calls
- optional PDF/public web sources for regulatory or security context

For ISPS-related context, prefer official public pages through URLs rather than unofficial full-text PDFs.

## Setup

Recommended Python version: 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` only if you want to build or query retrieval evidence. Deterministic analytics and carbon processing are designed to remain separate from the LLM path.

## Build the Demo Pipeline

One-command path:

```bash
./run_demo_pipeline.sh
```

Manual path:

```bash
python -m src.predict.data_prep \
  --traffic_csv data/PRJ912.csv \
  --traffic_csvs data/PRJ896.csv \
  --out_dir data/processed

python -m src.kpi.build_kpis \
  --traffic_csv data/PRJ912.csv \
  --traffic_csvs data/PRJ896.csv \
  --out_dir data/processed

python -m src.carbon.build \
  --processed_dir data/processed \
  --out_dir data/processed
```

Optional model and retrieval steps:

```bash
python -m src.predict.train_destination --training_rows data/processed/training_rows.parquet --model_dir models
python -m src.predict.train_eta --training_rows data/processed/training_rows.parquet --model_dir models
python -m src.predict.anomaly --training_rows data/processed/training_rows.parquet --model_dir models

python -m src.index.build_index \
  --traffic_csv data/PRJ912.csv \
  --traffic_csvs data/PRJ896.csv \
  --persist_dir data/chroma
```

## Run the Review UI

```bash
./run_streamlit.sh
```

The UI includes:
- ask interface with deterministic routing
- evidence and method-step panels
- confidence and coverage notes
- charts for supported result types
- strict state handling for unsupported or non-computable carbon questions

## Demo Questions

- `How many vessels arrived at LUBECK in March 2022?`
- `Is Friday usually busier than Monday at LVVNT?`
- `What will congestion look like next Friday at LUBECK?`
- `Why was LVVNT congested on 2021-01-01?`
- `Any unusual spikes in arrivals at GDANSK in 2021-02?`
- `What are TTW emissions at SEGOT in March 2022 for CO2e, NOx, SOx, and PM?`
- `Show WTW CO2e emissions at LVVNT between 2022-02-01 and 2022-02-28.`

## Data Quality and Answer Boundaries

Supported well:
- arrival volume, busiest periods, dwell proxy, congestion proxy, historical-pattern forecasts
- TTW pollutants and WTW `CO2e` where required AIS and port-call data is available

Not supported as ground truth unless additional data is provided:
- berth crane utilization
- gate queue length
- TEU throughput
- yard occupancy from terminal operating systems

For non-computable states, the UI suppresses numeric emission cards instead of showing false zero values. Retrieved evidence and deterministic results are presented separately.

## Cloud and Deployment Notes

The repository includes Streamlit and FastAPI deployment paths, plus optional bundle bootstrap settings for cloud environments:
- `APP_PROCESSED_BUNDLE_URL`
- `APP_EVENTS_BUNDLE_URL`
- `APP_CHROMA_BUNDLE_URL`
- `APP_CHROMA_MANIFEST_URL`
- `VECTOR_DB_MODE=remote` with `CHROMA_*` settings

Full local retrieval parity can require a large Chroma store, so lightweight hosted demos may use bundled processed data or remote vector storage.

## Tests

Run unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run forecast backtest:

```bash
python -m src.forecast.backtest --processed_dir data/processed
```

## Product Management Relevance

This project demonstrates:
- translating operational data problems into API and UI capabilities
- separating deterministic analytics from generative evidence
- documenting supported, unsupported, and data-quality-limited questions
- designing inspectable outputs for cross-functional review
- exposing service boundaries that could be standardized for internal product teams
