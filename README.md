# SupplySync AI

**Inventory decision system that combines a trained LightGBM forecaster with classical supply-chain methods, served through a FastAPI backend and a Next.js dashboard.**

Built on the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (~1M transactions, ~4,900 SKUs).

**Docs:** [guided tour](docs/PROJECT_GUIDED_TOUR.md) � [project explained](docs/PROJECT_EXPLAINED.md) � [architecture](docs/architecture.md) � [API reference](docs/api.md)

---


## Production Deployment

Status: **Live**

| Surface | Provider | URL / Notes |
|---|---|---|
| Live application | Vercel | https://supplysync-ai.vercel.app/ |
| Backend API | Render | https://supplysync-ai.onrender.com |
| Health endpoint | Render | https://supplysync-ai.onrender.com/health |
| Database | Neon PostgreSQL | Alembic-managed PostgreSQL schema |
| Runtime ML artifacts | GitHub Releases | `v1.0-artifacts` bundle |

The deployed stack is a Next.js frontend on Vercel calling a FastAPI backend on
Render, backed by Neon PostgreSQL. Runtime model/data artifacts are intentionally
not committed to Git: the trained LightGBM `.pkl` file and processed demand
parquet are supplied through the versioned GitHub Release artifact bundle.

The live decision workflow uses the same project components documented below:
LightGBM forecasting, Croston-SBA/statistical fallbacks, demand-pattern routing,
reorder point + safety-stock inventory logic, residual forecast-error uncertainty,
and forecast-evaluation/evidence-routing infrastructure. Free-tier hosting may
cold start after inactivity.

---

## Demo quickstart

Place `online_retail_II.csv` at `data/raw/online_retail_II.csv` (see [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii)), then:

```bash
make demo          # installs deps, trains the model, computes KPIs
make backend       # shell 1: http://localhost:8000
make frontend      # shell 2: http://localhost:3000
```

That's it: open `http://localhost:3000` and follow the demo script in [`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md). The system runs with auth off by default so the dashboard opens immediately; flip to demo-auth with `AUTH_MODE=demo` + `DEMO_PASSWORD=...` when you want the login gate.

---

## Screenshots

> Capture these yourself from a running local instance — see [`docs/screenshots/README.md`](docs/screenshots/README.md) for a 10-minute walkthrough. Files land in `docs/screenshots/` and the tags below resolve automatically.

| | |
|---|---|
| ![Dashboard](docs/screenshots/01-dashboard.png) | ![SKU detail — ML path](docs/screenshots/02-sku-ml-path.png) |
| **Dashboard** — KPI cards, per-SKU recommendations with provenance badges, row-level risk colour bars, SQLAlchemy-persisted activity panel. | **SKU detail (ML path)** — regular-demand SKU, LightGBM forecast, explanation trio, historical chart with demand-reference overlays. |
| ![SKU detail — synthetic path](docs/screenshots/03-sku-synthetic-path.png) | ![Sliders drive the recommendation](docs/screenshots/04-sliders-rerun.png) |
| **Unknown-SKU fallback** — amber synthetic-data banner, statistical method, provenance preserved end-to-end. | **Live assumption sliders** — lead-time and service-level changes re-run `/api/analyze` and update the decision in-place. |
| ![User-editable stock input](docs/screenshots/05-stock-override.png) | |
| **User-editable current stock** — type real inventory, the analysis re-runs, and the value is saved through the backend stock API with a browser fallback only for degraded/offline states. | |

---

## Headline result (Online Retail II demo)

Measured via `scripts/compute_kpis.py` on the top-10 SKUs by total demand, simulating 30-day policies with `service_level=0.95`, `lead_time=7`:

- **37.8% cost reduction** vs. a naive fixed-threshold reorder policy
- **95.7% fill rate** across the simulated SKUs
- **169 passing backend tests** with 6 intentional skips, covering forecasting routing, inventory math, evaluation, simulation, API contracts, provenance, auth, persistence, and cross-SKU generalization.

> These numbers are specific to UCI Online Retail II's top-10 SKUs. For a fair read on whether this approach fits **your** data, see [`docs/bring-your-own-data.md`](docs/bring-your-own-data.md) and run `python backend/scripts/evaluate_cross_sku.py`.

---

## Overview

SupplySync turns raw retail sales into actionable reorder recommendations. For any SKU it:

1. **Classifies** the demand pattern — *regular*, *intermittent*, or *highly intermittent*.
2. **Forecasts** demand with the method best suited to that pattern (LightGBM, Croston's, or a conservative buffer) and falls back to a simple moving average only when the ML path is not usable.
3. **Sizes safety stock** using dynamic rolling forecast error or a traditional Z × σ × √L formula.
4. **Computes a reorder point** and quantity, then applies business constraints (MOQ, order multiple, max-order cap, optional cross-SKU budget optimization).
5. **Surfaces data provenance** end-to-end so the UI can always tell the user whether a value came from recorded data, the model, a statistical method, a rule-based fallback, or a demo placeholder.

The project is deliberately scoped as a **portfolio-grade demo**: it runs on a single real dataset, returns honest provenance, and exposes its fallbacks instead of hiding them.

---

## Key Features

- **Adaptive forecasting** — one pipeline, three methods, picked per SKU based on demand sparsity. Regular → LightGBM; intermittent → Croston SBA; highly intermittent → conservative buffer.
- **Evidence-based routing with safe defaults** — when matching evaluation evidence is sufficient, recent, horizon-compatible, and materially better, the router can choose a stronger existing method; otherwise it keeps the demand-pattern default.
- **Trained LightGBM model drives live inference** — the API builds a training-shaped feature row from the latest history at request time and feeds it through the trained model; it does not silently degrade to an average.
- **Reproducible model identity** — the LightGBM artifact has an explicit version, SHA-256 checksum, feature-schema version, lifecycle status, and prediction-log linkage through `model_artifacts`.
- **Explicit provenance** — every `/api/analyze` response includes `demand_source` (`historical | request | synthetic`) and `forecast_source` (`model_forecast | statistical_method | rule_based_estimate | unavailable`). The UI renders matching badges on every surface.
- **User-editable current stock** — type real stock into any SKU, watch the recommendation and risk band change, and persist values through server-side stock endpoints with a browser fallback for degraded/offline states.
- **Configurable assumptions** — lead time (1–30 d) and service level (51–99%) sliders on the SKU detail page re-run `/api/analyze` in place.
- **Bring-your-own-data (CLI)** — run training, evaluation, and baselines on any retail CSV via `DATA_CSV_PATH` + optional column mapping. See [`docs/bring-your-own-data.md`](docs/bring-your-own-data.md).
- **Cross-SKU generalization eval** — `scripts/evaluate_cross_sku.py` holds out SKUs the model never saw and reports LightGBM vs. baseline performance fold-by-fold. The honest answer to "does it generalize?"
- **Real historical chart** — the SKU detail page fetches actual recorded daily demand from `/api/skus/{sku}/history`. When history isn't available, the chart is replaced by an honest "data unavailable" panel instead of fabricated bars.
- **Business constraints** — MOQ, order multiples, max order cap, and a greedy budget optimizer across SKUs.
- **Simulation-validated KPIs** — a day-by-day simulator compares naive vs. intelligent policies on real data to produce the savings/fill-rate numbers in the cached KPI JSON.
- **100+ backend tests** — reorder logic, business constraints, forecasting paths, simulation, history endpoint, provenance classification, model-info contract, decision block, evaluation primitives, cross-SKU generalization, and column-mapping round-trips.

---

## Tech Stack

| Layer | Stack |
|---|---|
| Modeling | LightGBM, scikit-learn, pandas, numpy, scipy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Next.js 16 (App Router), React 19, Tailwind CSS, Recharts |
| Data / persistence | UCI Online Retail II (CSV), processed to Parquet; PostgreSQL via SQLAlchemy/Alembic |
| Tooling | pytest (backend), TypeScript + ESLint (frontend), Docker Compose |

---

## Forecasting & Inventory Logic

### Demand classification

```
zero-demand share > 80%   → highly_intermittent
zero-demand share > 50%   → intermittent
otherwise                 → regular
```

### Forecast selection

| Pattern | Method | `forecast_source` label |
|---|---|---|
| regular | **LightGBM** (15 features: 7 daily lags, rolling mean/std windows, calendar) | `model_forecast` |
| intermittent | **Croston's method** with Syntetos-Boylan bias correction | `statistical_method` |
| highly intermittent | **Conservative forecast** (recent mean × 1.5 buffer) | `statistical_method` |
| regular but ML unavailable | 7-day simple moving average — **labeled as fallback** | `rule_based_estimate` |

The regular-demand ML path builds its single-row feature frame from the latest history using the same `create_lag_features` + `create_time_features` utilities used at training time, so the model sees identical preprocessing.

### Reorder decision

```
lead_time_demand = sum(forecast[:L])
safety_stock     = dynamic (rolling forecast error) OR Z × σ × √L
reorder_point    = lead_time_demand + safety_stock
order_qty        = max(0, reorder_point - current_stock)
```

Order quantity is then passed through supplier constraints (MOQ → order-multiple rounding → max-order cap) and optionally re-balanced across SKUs under a budget cap by a greedy priority allocator.

### Risk labels

The API computes a simple risk bucket from current stock vs. the P50/P90 of the demand series used for the analysis:

- stock < P50 → **HIGH**
- P50 ≤ stock < P90 → **MEDIUM**
- stock ≥ P90 → **LOW**

---

## Forecast Evaluation

`scripts/evaluate_forecast.py` backtests the forecasting stack against four
simple baselines on the top-20 SKUs, using a 30-day temporal holdout and
one-step-ahead predictions that feed the real previous-day actuals back at
each step. Results are saved to
[backend/data/forecast_evaluation.json](backend/data/forecast_evaluation.json)
and `forecast_evaluation.csv`.

Metrics used (intentionally not MAPE — it explodes on sparse demand):

- **MAE** — mean absolute error (units of demand)
- **RMSE** — penalizes large misses
- **bias** — mean(pred − actual); negative = systematic under-forecast
- **WAPE** — `sum|err| / sum|actual|`; the demand-weighted version of MAPE
- **MASE** — MAE scaled by the in-sample naive-repeat MAE; < 1 means the model beats repeating yesterday's value

Current results on the committed model (600 test points across 20 SKUs):

### All SKUs

| Model | MAE | RMSE | bias | WAPE | MASE |
|---|---:|---:|---:|---:|---:|
| naive_last | 109.92 | 175.12 | 5.82 | 1.19 | 0.97 |
| seasonal_naive_7 | 92.80 | 166.21 | 5.59 | 1.00 | 0.82 |
| moving_avg_7 | 86.40 | 129.38 | 2.25 | 0.93 | 0.76 |
| **croston_sba** | **80.41** | **125.88** | −1.46 | **0.87** | **0.71** |
| lightgbm | 91.67 | 143.58 | 26.06 | 0.99 | 0.83 |

### Regular-demand SKUs (510 test points)

| Model | MAE | RMSE | bias | WAPE | MASE |
|---|---:|---:|---:|---:|---:|
| naive_last | 119.80 | 189.96 | 7.14 | 1.17 | 1.07 |
| seasonal_naive_7 | 99.69 | 179.15 | 7.46 | 0.97 | 0.90 |
| moving_avg_7 | 93.69 | 139.93 | 3.34 | 0.92 | 0.84 |
| **croston_sba** | **87.57** | **136.40** | −0.88 | **0.86** | **0.78** |
| lightgbm | 96.30 | 152.34 | 24.39 | 0.94 | 0.89 |

### Intermittent SKUs (90 test points)

| Model | MAE | RMSE | bias | WAPE | MASE |
|---|---:|---:|---:|---:|---:|
| naive_last | 53.91 | 91.06 | −1.67 | 1.46 | 0.40 |
| seasonal_naive_7 | 53.72 | 92.90 | −4.99 | 1.46 | 0.40 |
| moving_avg_7 | 45.07 | 69.57 | −3.97 | 1.22 | 0.33 |
| **croston_sba** | **39.84** | **66.25** | −4.80 | **1.08** | 0.28 |
| lightgbm | 65.40 | 93.96 | 35.52 | 1.77 | 0.46 |

### What this actually says

- **MASE < 1 for every non-naive model** — all three classical baselines plus the LightGBM beat the repeat-yesterday baseline on this dataset.
- **On regular SKUs the LightGBM is competitive but not dominant**: it improves over `naive_last` by ~20% MAE and edges `seasonal_naive_7`, but a 7-day moving average and Croston-SBA both beat it on MAE / WAPE / MASE. That's honest: 20 SKUs and 13K training rows aren't enough for a tree model to consistently outperform strong statistical baselines on general retail demand.
- **On intermittent SKUs the LightGBM actively hurts** — bias of +35.5 means it systematically over-forecasts when demand is sparse. This validates the live routing: the analyze endpoint never sends intermittent SKUs through the model; it routes them to Croston (for `intermittent`) or a conservative buffer (for `highly_intermittent`), which is the best-performing option on that slice.
- **Croston-SBA is the best per-class model** and it's already wired into the live path for the SKU classes where it wins.

Reproduce with:

```bash
cd backend && python scripts/evaluate_forecast.py
```

## Architecture

```
data/raw/online_retail_II.csv
       │
       ▼  ingestion/load_retail_data.py
data/processed/daily_demand.parquet
       │
       ├──► features/ (lag_features, time_features, inference_features)
       │         │
       │         ▼
       │   scripts/train_model.py → saved_models/lightgbm_demand_forecast.pkl
       │                            + metadata.json (feature schema)
       │
       ▼  services/
    DataService            ← reads the processed parquet
    IntelligentInventoryService
       │
       ├─ adaptive_forecasting_service → forecasting/forecast_service (LightGBM)
       │                                 croston / conservative fallbacks
       ├─ inventory/reorder_point + business_constraints
       └─ uncertainty/ (dynamic sigma, prediction intervals)
       │
       ▼
    FastAPI (backend/main.py)
       │
       ▼
    Next.js dashboard (frontend/app)
```

---

## Data Flow for `POST /api/analyze`

1. Resolve demand: use the request body's `demand_history` if provided, else pull recorded history from `DataService`, else fall back to a deterministic synthetic Poisson series **tagged `demand_source="synthetic"`**.
2. Classify the demand pattern.
3. For *regular* SKUs, build the training-shaped feature row from the recent history and call the loaded LightGBM model recursively for the lead-time horizon. If history is too short or the model isn't available, emit `forecast_method="simple_average"` with `forecast_source="rule_based_estimate"`.
4. For *intermittent* / *highly intermittent* SKUs, run Croston's method or the conservative forecast — never the ML model.
5. Compute safety stock, reorder point, quantity, and apply supplier constraints.
6. Return the decision with both `demand_source` and `forecast_source` so the UI can label it truthfully.

---

## API Reference

All `/api/*` routes accept an optional `X-API-Key` header. If the `API_KEY` env var is unset, the key check is skipped (dev mode). `/health` is always public.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + readiness flags |
| GET | `/api/model-info` | Trained-artifact metadata + evaluation pointer |
| GET | `/api/kpis` | Simulated KPIs with interpretation metadata |
| GET | `/api/skus` | Top-20 SKU codes |
| GET | `/api/skus/details` | Top-20 SKUs with avg/total demand |
| GET | `/api/skus/{sku}/history?days=30` | Recorded daily demand for a SKU, or `available: false` if the SKU isn't in the processed dataset |
| POST | `/api/analyze` | Risk + reorder recommendation for a SKU, with provenance, decision, and explanation blocks |
| GET | `/api/analyses/recent?limit=N` | Most-recent persisted analyses from SQLAlchemy `analysis_runs` |
| GET | `/api/auth/status` | Public auth-mode probe (used by the frontend gate) |
| POST | `/api/auth/login` | Demo-mode login: `{username, password}` → session cookie |
| POST | `/api/auth/logout` | Clears the session cookie |
| GET | `/api/auth/me` | Current caller identity (session or API key) |

`POST /api/analyze` accepts optional `lead_time_days` (1–90) and `service_level` (0.5–1.0) overrides — the SKU detail page uses these for the live Assumptions panel.

Full request/response examples are in [docs/api.md](docs/api.md).

### Example — `POST /api/analyze`

Request:
```json
{ "sku": "85099B", "current_stock": 50 }
```

Response (abridged):
```json
{
  "sku": "85099B",
  "risk": "MEDIUM",
  "risk_color": "#eab308",
  "forecast": { "p50": 18.4, "p90": 42.0, "daily": [19.1, 18.8, 20.2, 17.6, 21.0, 18.9, 19.4] },
  "current_stock": 50,
  "recommended_order": 74,
  "action": "PURCHASE",
  "demand_pattern": "regular",
  "forecast_method": "ml_lightgbm",
  "demand_source": "historical",
  "forecast_source": "model_forecast"
}
```

For an unknown SKU, `demand_source` becomes `"synthetic"` and the UI shows an explicit "demo data" banner.

### Example — `GET /api/skus/85099B/history?days=5`

```json
{
  "sku": "85099B",
  "available": true,
  "history": [
    { "date": "2011-12-05", "demand": 18 },
    { "date": "2011-12-06", "demand": 24 },
    { "date": "2011-12-07", "demand": 12 },
    { "date": "2011-12-08", "demand": 0 },
    { "date": "2011-12-09", "demand": 31 }
  ]
}
```

For an unknown SKU: `{ "sku": "...", "available": false, "history": [] }`.

---

## Frontend

A single-page Next.js dashboard plus a per-SKU detail view. Everything that can be mistaken for real operational data is labeled.

- **Dashboard** (`/`) — pipeline diagram, KPI cards, SKU table. The Stock column carries a **DEMO** pill explaining that stock levels are derived from each SKU's average demand for illustration. The Method column shows the concrete forecast method and a colored provenance badge (*model*, *statistical*, or *rule-based*).
- **SKU detail** (`/sku/[id]`) — historical-demand bar chart fetched from `/api/skus/{sku}/history` (date-aware x-axis; honest empty state when no history is recorded), a recommendation card with a forecast-source badge, and an amber banner on synthetic-data paths reading *"Demo data for unknown SKU."*

### Provenance vocabulary (consistent across API and UI)

| Kind | Badge | Where |
|---|---|---|
| Real historical demand | **Recorded** | chart header |
| Caller-supplied demand | **User input** | chart header |
| Fabricated demand fallback | **Synthetic demo** | chart header + page-top banner |
| LightGBM prediction | **Model forecast** | recommendation card, dashboard row |
| Croston / conservative | **Statistical** | recommendation card, dashboard row |
| 7-day average fallback | **Rule-based** | recommendation card, dashboard row |
| Derived demo placeholder (stock levels) | **Demo value** / **DEMO** | stock field + column header |
| Missing from backend | **Unavailable** | anywhere applicable |

---

## Repository Structure

```
supplysync-ai/
├── backend/
│   ├── main.py                    # FastAPI app, lifespan, routes
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── bootstrap.py           # One-shot setup (train + KPIs)
│   │   ├── check_setup.py         # Reports on required artifacts
│   │   ├── train_model.py         # Builds daily_demand.parquet + trains LightGBM
│   │   ├── compute_kpis.py        # Inventory-policy simulation → cached_kpis.json
│   │   └── evaluate_forecast.py   # Backtests model vs. baselines → forecast_evaluation.{json,csv}
│   ├── src/
│   │   ├── services/              # DataService, IntelligentInventoryService, ModelService, adaptive_forecasting_service
│   │   ├── inventory/             # Reorder point, business constraints
│   │   ├── forecasting/           # Recursive LightGBM forecast service
│   │   ├── evaluation/            # Forecast metrics (MAE/RMSE/WAPE/MASE/bias) and baselines
│   │   ├── features/              # Lag, calendar, and inference-feature builders
│   │   ├── ingestion/             # Retail II loader
│   │   ├── simulation/            # Day-by-day simulator + policy comparison
│   │   └── uncertainty/           # Rolling sigma, prediction intervals, KPI utils
│   ├── saved_models/              # LightGBM artifact + feature-schema metadata (after training)
│   ├── data/cached_kpis.json      # Populated by compute_kpis.py
│   └── tests/                     # 47 pytest cases
├── frontend/
│   ├── app/                       # Next.js App Router (page.tsx, sku/[id]/page.tsx, api/)
│   ├── components/                # DataSourceBadge + UI primitives
│   ├── lib/api.ts                 # Typed API client
│   └── package.json
├── data/
│   ├── raw/online_retail_II.csv   # Not included — download from UCI
│   └── processed/daily_demand.parquet   # Produced by train_model.py
├── docker-compose.yml
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+ (tested on 3.12)
- Node.js 20+
- ~500 MB disk for the raw CSV + processed parquet + trained model

### 1. Get the dataset

Download `online_retail_II.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it at `data/raw/online_retail_II.csv`. This file is intentionally not committed.

### 2. Backend

```bash
cd backend
pip install -r requirements.txt

# See what is already present vs what needs generating
python scripts/check_setup.py

# One shot: builds data/processed/daily_demand.parquet, trains the LightGBM
# model, and populates backend/data/cached_kpis.json. Safe to re-run; only
# generates what is missing. Use --force to regenerate everything.
python scripts/bootstrap.py

# Run the tests (no artifacts required — tests use in-memory stubs)
python -m pytest tests/

# Start the API
uvicorn main:app --reload --port 8000
```

Individual steps, if you prefer running them separately:

```bash
python scripts/train_model.py        # parquet + LightGBM pkl + metadata
python scripts/compute_kpis.py       # inventory backtest → cached_kpis.json
python scripts/evaluate_forecast.py  # model vs. baselines → forecast_evaluation.{json,csv}
```

Model artifacts are generated as candidates, not silently promoted. To register
and promote a validated artifact:

```bash
python scripts/register_model_artifact.py --model-name lightgbm_demand_forecast
python scripts/promote_model.py --artifact-id <id>
python scripts/promote_model.py --artifact-id <id> --force
```

Once running, `curl http://localhost:8000/health` reports which artifacts are loaded:

```json
{
  "status": "online",
  "model_loaded": true,
  "data_available": true,
  "kpis_available": true,
  "hint": null
}
```

If any field is `false`, `hint` explains how to fix it.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
# open http://localhost:3000
```

The frontend reads `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). To override, copy `.env.example` to `.env.local` and edit.

### Docker

```bash
# Host-side prerequisites: the CSV must exist at data/raw/online_retail_II.csv
# Generate parquet + model + KPIs on the host first (they're bind-mounted in):
cd backend && python scripts/bootstrap.py && cd ..

docker-compose up --build
```

The compose file bind-mounts `./backend` at `/app` and the dataset at `/data` (read-only). The backend's `/health` endpoint is wired to a Docker healthcheck, so the frontend container waits until the API is serving before starting.

### Environment variables

**Backend** (`backend/.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `backend/saved_models` | Where `ModelService` loads/saves artifacts |
| `EVIDENCE_ROUTING_ENABLED` | `false` | Enable evidence-based method selection. Disabled by default until logged evidence matures. |
| `ROUTING_PRIMARY_METRIC` | `wape` | Primary lower-is-better routing metric. Avoids naive MAPE on sparse demand. |
| `ROUTING_MIN_EVALUATION_POINTS` | `30` | Minimum evaluated points required before evidence can affect routing. |
| `ROUTING_MIN_RELATIVE_IMPROVEMENT` | `0.05` | Required relative error improvement before switching away from the default method. |
| `ROUTING_EVIDENCE_LOOKBACK_DAYS` | `365` | Ignore evaluation evidence older than this window. |
| `LEAD_TIME_DAYS` | `7` | Default lead time (analyze can override per request) |
| `SERVICE_LEVEL` | `0.95` | Default service level (analyze can override per request) |
| `AUTH_MODE` | `off` | `off` (open API) or `demo` (session-cookie login required) |
| `DEMO_USER` / `DEMO_PASSWORD` | `demo` / *(unset)* | Shared credential when `AUTH_MODE=demo` |
| `SESSION_SECRET` | *(ephemeral)* | HMAC key for signing session cookies |
| `API_KEY` | *(unset)* | Legacy header auth — honored in either mode when set |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `false` | Logging configuration parsed by the backend settings layer |
| `DATABASE_URL` | `sqlite:///backend/data/supplysync.db` | SQLAlchemy database URL. Docker/production should use PostgreSQL. |

Backend runtime settings are composed in `backend/src/config/settings.py`, with
domain-specific modules for app, auth, database, forecasting, inventory, and
logging settings. Invalid numeric settings fail fast during startup instead of
silently falling back.

**Frontend** (`frontend/.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend base URL |

---

## Tests

```bash
cd backend && python -m pytest tests/
# 100+ passed across 14 files
```

Coverage breakdown:

- **Forecasting** (19) — demand classification, Croston, conservative, adaptive routing, trained-model path, fallback reasons, short-history safety, end-to-end service decision.
- **Evaluation** (17) — MAE / RMSE / bias / WAPE / MASE correctness, denominator-handling for sparse actuals, baseline predictors and walk-forward windowing.
- **Inventory logic** (15) — reorder decisions, dynamic vs. traditional safety stock, business constraints, budget optimizer.
- **Simulation** (5) — stockouts, holding cost, combined-cost invariants.
- **API shapes + provenance** (9) — `/api/skus/{sku}/history` shape and empty-state, `/api/analyze` provenance fields across historical / synthetic / request paths, method-to-source classification, `/health` readiness contract.
- **Model-info contract** (5), **decision block + persistence** (14), **configurable inputs** (7), **auth** (9), **artifact hygiene** (3).
- **Cross-SKU generalization** (3) — column-mapping round-trips, unseen-SKU fallback path, cross-SKU evaluation output shape.

Frontend checks:

```bash
cd frontend
npm test          # tsc --noEmit + eslint
npm run build     # production Next.js build (runs in CI too)
```

---

## Bring your own data

The demo runs on UCI Online Retail II, but the training, evaluation, and forecasting logic aren't coupled to that specific CSV. Full walkthrough in [`docs/bring-your-own-data.md`](docs/bring-your-own-data.md). The three entry points:

```bash
# 1) Train on your CSV (same UCI column names)
DATA_CSV_PATH=/path/to/my_sales.csv python backend/scripts/train_model.py

# 2) Different column names? Pass a mapping JSON.
python backend/scripts/evaluate_custom_dataset.py \
    --csv /path/to/my_sales.csv \
    --column-mapping column_mapping.json

# 3) Unseen-SKU generalization test (the honest "does it generalize?" answer)
python backend/scripts/evaluate_cross_sku.py --folds 5
```

Cross-SKU evaluation holds out SKUs the model never saw and reports LightGBM vs. four baselines fold-by-fold. The full JSON output lands at `backend/data/cross_sku_evaluation.json`.

---

## Demo vs. real — what's honest about this project

Split explicitly so nothing is oversold:

| Surface | What's real | What's demo |
|---|---|---|
| **Forecasting** | LightGBM + Croston + moving-average fallback fully implemented; invoked on every `/api/analyze` call; fallbacks are labeled, never hidden. | Trained on top-20 SKUs only. |
| **Model lifecycle** | LightGBM artifact metadata includes checksum, version, feature schema, and lifecycle status; prediction logs keep the exact `model_artifact_id` used. | Promotion is command-line/manual, not a full model registry platform. |
| **Evaluation** | MAE / RMSE / WAPE / MASE correctly computed vs. four baselines on a temporal holdout. Cross-SKU script reports the unseen-SKU story honestly (even if baselines win). | Evaluation baseline set is fixed (naive / seasonal / MA / Croston); no bigger models compared. |
| **Inventory math** | Z-score safety stock, reorder point, MOQ + order-multiple + max-order constraints, dynamic sigma from rolling residuals. Same math on every request. | — |
| **Current stock** | User can type real stock per SKU; re-runs the analysis; persists through backend stock endpoints. | Default values are demo-derived until a real stock update exists. Browser storage remains only as a degraded/offline fallback. |
| **Auth** | HMAC-signed session cookie, configurable demo password. | Demo-mode only — no OAuth, no multi-tenant, no proper RBAC. |
| **Persistence** | `POST /api/analyze` writes every decision and linked prediction log through SQLAlchemy repositories; the Recent analyses panel reads `analysis_runs`. Docker/CI exercise PostgreSQL migrations. | Local default remains SQLite for lightweight development unless `DATABASE_URL` points at PostgreSQL. |
| **Data ingestion** | Bring-your-own-CSV with column-mapping config for real retail feeds. | No live inventory integration, no ERP connectors. |

---

## Current State & Limitations

This is a **portfolio-grade demo**, not a production ERP. Things that are honest to know up-front:

- **Stock persistence is live but demo-scoped.** Edited stock values go through server-side stock endpoints backed by PostgreSQL. The frontend still keeps a browser fallback for offline/error states, and there is no ERP/WMS integration.
- **Only 20 SKUs in training.** The full UCI dataset has ~4,900 SKUs; we train on the top-20 by total demand to keep the demo fast. Cross-SKU evaluation covers generalization across that set but doesn't extrapolate to cold-start products.
- **Headline metrics are dataset-specific.** The 37.8% cost reduction / 95.7% fill rate come from simulating policies on UCI Online Retail II's top-10 SKUs. Expect different numbers on your data — that's why the cross-SKU and custom-dataset evaluation paths exist.
- **The trained model and processed parquet aren't committed** (only the metadata JSON is). Run `python scripts/bootstrap.py` once after cloning to generate both. Until they exist, the regular-demand path falls back to `simple_average` with `forecast_source="rule_based_estimate"`, `/api/skus*` return 503 with a `hint`, and `/health` reports `model_loaded: false` / `data_available: false`.
- **Model accuracy is modest.** The trained LightGBM reports MAE ≈ 95 units on the held-out 30-day split — credible for a demo trained on public retail data, not state-of-the-art. Classical baselines (Croston, moving-average) beat it on intermittent SKUs; the live router handles that routing automatically.
- **Scope is intentional.** Single product family (general retail), no background job scheduler, no multi-tenant isolation, no KPI drill-downs, no backorder or multi-sourcing logic.
- **Frontend demo signals are deliberate.** The "Demo value" pill, server/offline stock-source labels, and amber "synthetic demo" banner exist so a viewer can never mistake illustrative values for real operational data.

---

## Future Improvements

- Keep PostgreSQL migrations validated in CI and decide whether old local SQLite analysis snapshots need a one-time import.
- UI upload flow for bring-your-own-data (currently CLI-only).
- Probabilistic forecasting (quantile regression or conformal intervals) end-to-end, replacing the current residual-based approximation.
- Per-SKU model selection via cross-validated AIC/MAPE rather than a sparsity threshold.
- Background training / KPI refresh jobs so cached artifacts stay current.
- Richer explanations in the UI (feature attributions from the LightGBM model).
- Per-user auth, multi-tenant data isolation, and a proper deployment target.

---

## Why This Project

A recruiter or engineer can read the code and verify that:

- The **ML path is wired end-to-end** — features are built at inference time to match training, the model is actually invoked, and fallbacks are explicit, logged, and surfaced in the API.
- The **classical supply-chain layer** is correct — Croston with SBA correction, Z-score safety stock, dynamic sigma, MOQ/multiple/max-order constraints, budget-aware greedy allocation.
- **Data provenance is taken seriously** — real vs. model vs. statistical vs. rule-based vs. demo is distinguishable in every response and visible on every surface.
- **The numbers are reproducible** — `scripts/compute_kpis.py` regenerates the cached KPIs from the same simulator the tests exercise.
- **Generalization is inspected honestly** — `scripts/evaluate_cross_sku.py` holds out SKUs the model never saw and reports the result, even when baselines win.
- **Tests hold the contract** — 100+ pytest cases cover routing between ML/statistical/fallback paths, provenance labels, history endpoint shape, simulation invariants, reorder/constraint logic, auth, cross-SKU generalization, and column-mapping round-trips.

It's meant to be honest about what it is — a small, well-scoped, well-labeled inventory decision-support demo — and straightforward to extend toward something more production-grade.

---

## License

MIT

## Architecture Update: Analysis Service

`POST /api/analyze` now delegates orchestration to
`backend/src/services/analysis_service.py`. The FastAPI route keeps the same
request and response contract, while `AnalysisService` owns demand resolution,
synthetic fallback, inventory-service orchestration, deterministic
explanations, SQLAlchemy-first analysis persistence, and linked prediction-log
creation.

Recent analysis history reads SQLAlchemy `analysis_runs`. Stock persistence is server-side through the existing `StockService` and `StockRepository` path, with the frontend retaining a browser fallback only for degraded demo use.

## Logged Prediction Evaluation

SupplySync now separates two kinds of forecast evidence:

- offline model benchmarking, generated by scripts such as
  `backend/scripts/evaluate_forecast.py`;
- logged prediction evaluation, generated after persisted `/api/analyze`
  forecasts have a completed actual-demand window.

The logged workflow is:

```text
analysis_runs
  -> prediction_logs
  -> actual demand from DataService
  -> forecast_evaluations
```

Run it with:

```bash
cd backend
python scripts/evaluate_logged_predictions.py
```

The evaluator skips predictions whose horizon has not completed, predictions
without available recorded actuals, and predictions that already have an
evaluation row.

## Evidence-Based Routing

Forecast routing is now evidence-aware but still conservative:

```text
demand history
  -> demand pattern
  -> eligible existing methods
  -> matching evaluation evidence
  -> safe default unless the evidence is strong enough
```

The primary metric is WAPE. Evidence must match the demand pattern and forecast
horizon, meet `ROUTING_MIN_EVALUATION_POINTS`, be within
`ROUTING_EVIDENCE_LOOKBACK_DAYS`, and beat the default method by at least
`ROUTING_MIN_RELATIVE_IMPROVEMENT`.

Logged SQLAlchemy evaluations are preferred. Offline benchmark evidence from
`backend/data/forecast_evaluation.json` is used only as bootstrap evidence when
the horizon matches. The committed offline artifact is a 30-day evaluation, so
it does not silently change the default 7-day `/api/analyze` behavior.

Preview the policy against the offline artifact with:

```bash
cd backend
python scripts/simulate_model_routing.py
```

## Model Artifact Lifecycle

SupplySync treats the LightGBM file as a versioned artifact, not just a generic
pickle. `backend/saved_models/lightgbm_demand_forecast_metadata.json` records
the model version, SHA-256 checksum, feature-schema version/checksum, training
summary, and lifecycle status.

At startup, `ModelService` validates the artifact path, checksum, model type,
feature schema version, and ordered feature columns. If validation fails, the
API keeps running and regular-demand SKUs use the existing `simple_average`
fallback instead of executing an incompatible model.

The database table `model_artifacts` is the lifecycle source of truth after an
artifact is registered. Historical `prediction_logs` keep their original
`model_artifact_id`, `model_version`, and `feature_schema_version`, even if a
new artifact is promoted later.


