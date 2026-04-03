# Eagle Eye Review Artifacts

This folder stores machine-readable UI/API review outputs for non-JS inspection.

## Local generation

From repository root:

```bash
./run_ui_review.sh
```

Outputs:

- `/Users/praharshchintu/Documents/New project/review/latest/review_index.json`
- `/Users/praharshchintu/Documents/New project/review/latest/review_summary.md`
- `/Users/praharshchintu/Documents/New project/review/latest/screenshots/*`

Run history:

- `/Users/praharshchintu/Documents/New project/review/runs/<run_id>/`

## Scenario fixture

- `/Users/praharshchintu/Documents/New project/review/scenarios.json`

Categories covered:

- Traffic descriptive
- Vessel investigation
- Forecast
- Carbon deterministic
- Carbon retrieval-only/no-data
- Unsupported scope

## CI publication

Workflow:

- `/Users/praharshchintu/Documents/New project/.github/workflows/ui-review.yml`

It publishes `review/latest` + `review/runs/<run_id>` to GitHub Pages under `/review/`.

