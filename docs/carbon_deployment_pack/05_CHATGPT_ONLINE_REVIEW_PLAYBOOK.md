# ChatGPT Online Review Playbook (Eagle Eye)

## Goal
Expose a live Eagle Eye URL plus source-of-truth artifacts so a ChatGPT agent can review implementation, run black-box checks, and validate benchmark claims.

## What ChatGPT needs
1. Public app URL (temporary tunnel URL).
2. GitHub repo URL.
3. Benchmark files:
   - `/Users/praharshchintu/Documents/New project/evaluation/thesis/results/summary.json`
   - `/Users/praharshchintu/Documents/New project/evaluation/thesis/results/per_query_results.csv`
4. Project status report:
   - `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/04_FULL_PROJECT_STATUS_BENCHMARK_DEPLOYMENT_2026-03-25.md`

## Runbook (recommended)

### 1) Start local app cleanly
```bash
cd "/Users/praharshchintu/Documents/New project"
pkill -f "streamlit run app/streamlit_app.py" || true
pkill -f "cloudflared tunnel --protocol http2 --url" || true
pkill -f "lt --port 8501" || true

source .venv/bin/activate
set -a; source .env; set +a
PYTHONPATH=. .venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
Keep this terminal open.

### 2) In a second terminal, start a public tunnel
```bash
cd "/Users/praharshchintu/Documents/New project"
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8501
```
Copy the printed `https://....trycloudflare.com` URL.

If cloudflared is blocked, fallback:
```bash
lt --port 8501 --local-host 127.0.0.1
```
Copy the printed `https://....loca.lt` URL.

### 3) Health checks before sharing
```bash
curl -s -o /dev/null -w 'LOCAL:%{http_code}\n' http://127.0.0.1:8501/_stcore/health
curl -s -o /dev/null -w 'PUBLIC:%{http_code}\n' <PUBLIC_URL>/_stcore/health
```
Expected: `LOCAL:200` and `PUBLIC:200`.

## Share package with ChatGPT
Provide this in one message:
1. App URL: `<PUBLIC_URL>`
2. Repo URL: `https://github.com/Praharsh-Projects/Eagle_Eye`
3. Ask ChatGPT to validate categories:
   - traffic descriptive
   - vessel investigation
   - congestion forecast
   - carbon deterministic
   - carbon retrieval-only/no-data semantics
   - unsupported-scope refusal
4. Ask ChatGPT to compare observed behavior against benchmark and contract files.

## If public URL fails
- Restart tunnel only (do not restart Streamlit first).
- Re-run public health check.
- If still failing, rotate to a new tunnel URL and resend.

## Important limitation
- This is a session-hosted demo URL: it stays alive only while your Mac, Streamlit process, and tunnel process remain running.
- Free tunnels do not guarantee stable domain names like `eagle-eye.*` without your own managed DNS/tunnel configuration.
