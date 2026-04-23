# Demo walkthrough

A ~5-minute script for showing the app in a portfolio review or recording a short video. Assumes you've run the setup below and have both the backend (`uvicorn main:app --port 8000`) and the frontend (`npm run dev`, port 3000) running.

### Fastest setup

```bash
make demo       # installs deps + trains the model + computes KPIs
make backend    # shell 1
make frontend   # shell 2
```

If you want the login gate during the demo:

```bash
export AUTH_MODE=demo
export DEMO_PASSWORD=letmein
make backend
```

No fake flourishes — every surface below reflects the real product behavior after Suggestions 1–17.

---

## Suggested flow

### 0. One-minute sanity check (terminal)

```bash
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/api/model-info | jq '{model_name, artifact_available, trained_at, evaluation}'
```

What to point at:
- `/health` shows `model_loaded: true`, `data_available: true`, `kpis_available: true`, `hint: null`.
- `/api/model-info` names the artifact, its train date, feature count, and whether the evaluation report exists.

### 1. Dashboard (`http://localhost:3000`) — 60 seconds

What to show, in order:
1. **Header pipeline**: *Ingest → Classify → Forecast → Optimize* — "this is the whole story on one line."
2. **KPI cards** (hover each): the tooltips come directly from `/api/kpis` → `interpretation.metric_meanings`, so the text explains what each number actually means and which baseline it's against.
3. **SKU table**:
   - Stock column has a small `DEMO` pill — demo values derived from average demand, not live inventory.
   - Method column shows both the concrete method (`ml_lightgbm`, `croston`, etc.) and a provenance badge (`MODEL FORECAST`, `STATISTICAL`, `RULE-BASED`) — the colors differ, so a reviewer can see at a glance which SKUs the ML is actually driving.
4. **"Recent analyses" panel** (below the table): every `/api/analyze` call is persisted in `backend/data/analyses.sqlite`. Clicking a row deep-links to that SKU.

### 2. SKU detail — click any SKU row — 2 minutes

Walk top-to-bottom:
1. **Summary strip** (one row above the grid): *Pattern · Forecast method + provenance badge · Input data + provenance badge · Action* — full context in 5 fields.
2. **"Why this forecast path"** card (three columns: *Classification*, *Method choice*, *Risk reasoning*) — every sentence is template-generated from the real inputs (zero-demand %, observed days, stock vs P50/P90). Point out the confidence caveat line.
3. **Historical demand chart** on the left — real recorded daily demand from `/api/skus/{sku}/history`, date-aware x-axis.
4. **Recommendation card** on the right — risk-bordered; reads *"Recommended order: N units"* with a one-sentence `decision.why` explaining the numbers.
5. **Key Metrics** — Current Stock carries the `DEMO` pill so there's no confusion about what's real.
6. **Model & Method** block — model name, type, artifact state (`loaded` / `not loaded`), training date, feature count, dataset, evaluation availability. If the `.pkl` isn't loaded, Type flips to `rule based fallback` and Artifact to `not loaded` in amber.
7. **"How This Decision Was Made"** — four dynamic steps (Classify, Forecast, Safety Stock, Decision) with actual numbers, not boilerplate.

### 3. Synthetic-fallback case — 30 seconds

Visit `/sku/NOT-A-REAL-SKU`. Point out:
- Amber top banner: *"Demo data for unknown SKU"*.
- Chart header badge flips to **Synthetic demo**.
- Everything downstream still computes, but the Why-this-forecast-path confidence caveat explicitly says the recommendation is illustrative only.

### 4. Artifact-missing case — optional 30 seconds

If you want to show the fallback visibly:
```bash
mv backend/saved_models/lightgbm_demand_forecast.pkl /tmp/  # hide the artifact
# restart uvicorn
```
Now the SKU detail Model & Method block shows Type: `rule based fallback`, Artifact: `not loaded`, both in amber. The summary strip flips Forecast to `simple_average [RULE-BASED]`.
```bash
mv /tmp/lightgbm_demand_forecast.pkl backend/saved_models/  # restore
```

### 5. API inspection (optional) — 45 seconds

```bash
curl -s http://localhost:8000/api/analyses/recent?limit=5 | jq '.items[] | {sku, created_at, action, forecast_method, recommended_order}'
curl -s -X POST http://localhost:8000/api/analyze \
    -H 'content-type: application/json' \
    -d '{"sku":"85099B","current_stock":50}' | jq '.explanation,.model_info'
```

What this proves:
- Results persist across calls (SQLite).
- Every response carries honest provenance + explanation, not just the final number.

---

## Screenshots

Drop PNG/JPG captures under [./screenshots](./screenshots) using the filenames listed in [./screenshots/README.md](./screenshots/README.md). Reference them from this doc when you add them:

```markdown
![Dashboard](./screenshots/dashboard.png)
```

---

## What *not* to demo

- The `/api/copilotkit` route — scaffolding only; not wired to the UI.
- The `app/api/copilotkit/route.ts` endpoint — same reason.
- The cached KPI numbers as if they were from a live simulation — they're from a one-time backtest run; reproduce with `python scripts/compute_kpis.py`.
