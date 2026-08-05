# SupplySync AI Guided Tour

This document is for someone who just cloned the repository and wants to
understand the project in about 30 minutes.

Read this before reading the source code. It explains what the system does,
where the important logic lives, how data moves through the app, and what to
debug first when something breaks.

## 0. The Short Version

SupplySync AI is an inventory decision-support system for retail SKUs.

It takes historical sales data, forecasts near-term demand, calculates whether
current stock is enough, and recommends whether to order more inventory. The
project combines:

- a FastAPI backend,
- a Next.js dashboard,
- LightGBM forecasting for regular-demand SKUs,
- Croston-SBA / conservative statistical methods for sparse demand,
- reorder-point inventory math,
- server-side stock persistence through SQLAlchemy repositories,
- local/demo analysis persistence through SQLite,
- reproducible evaluation and simulation scripts.

The project is intentionally a clean modular monolith. It is not a
microservice platform, not an ERP, and not a fake "AI wrapper." The strongest
engineering idea in the repo is that it labels what is real, what is inferred,
and what is demo fallback.

## 1. What Problem Are We Solving?

Inventory teams need to answer a practical question:

> "For this SKU, given recent demand and current stock, should we reorder?"

That question has three parts:

1. Forecast likely demand over the supplier lead time.
2. Add a safety buffer for uncertainty.
3. Compare the reorder point with current stock and apply business constraints.

SupplySync turns that into a reviewer-friendly workflow:

```text
historical sales
  -> demand pattern classification
  -> forecasting method selection
  -> safety stock calculation
  -> reorder-point calculation
  -> MOQ / order multiple / max-order constraints
  -> API response with provenance and explanation
  -> dashboard recommendation
```

The business problem is stockout prevention without blindly over-ordering. The
technical problem is wiring forecasting, inventory math, persistence, API
contracts, and frontend provenance into one understandable system.

## 2. What Should I Run First?

From the repository root:

```bash
make demo
make backend
make frontend
```

Then open:

```text
http://localhost:3000
```

If you only want to inspect backend readiness:

```bash
curl http://localhost:8000/health
```

For the newer SQLAlchemy-backed stock endpoints, run migrations from
`backend/` before testing stock writes manually:

```bash
python -m alembic upgrade head
```

The backend still accepts `current_stock` in `POST /api/analyze`, so the main
forecasting demo can run even while the stock database is being set up.

## 3. The 30-Minute Reading Path

Use this order if you are reviewing the project for the first time.

### First 5 minutes: Product and architecture

Read:

- [README.md](../README.md)
- [docs/architecture.md](./architecture.md)
- [docs/api.md](./api.md)

Goal: understand the product pitch, the main runtime flow, and the API surface.

### Next 10 minutes: Backend decision path

Read:

- [backend/main.py](../backend/main.py)
- [backend/src/services/intelligent_inventory_service.py](../backend/src/services/intelligent_inventory_service.py)
- [backend/src/services/adaptive_forecasting_service.py](../backend/src/services/adaptive_forecasting_service.py)
- [backend/src/inventory/reorder_point.py](../backend/src/inventory/reorder_point.py)
- [backend/src/inventory/business_constraints.py](../backend/src/inventory/business_constraints.py)

Goal: understand what happens when the API returns a recommendation.

### Next 5 minutes: Data and ML

Read:

- [backend/src/ingestion/load_retail_data.py](../backend/src/ingestion/load_retail_data.py)
- [backend/src/features/inference_features.py](../backend/src/features/inference_features.py)
- [backend/scripts/train_model.py](../backend/scripts/train_model.py)
- [backend/scripts/evaluate_forecast.py](../backend/scripts/evaluate_forecast.py)
- [backend/data/forecast_evaluation.json](../backend/data/forecast_evaluation.json)

Goal: understand where the data comes from, how features are built, and why the
system does not always use LightGBM.

### Next 5 minutes: Frontend and user journey

Read:

- [frontend/app/page.tsx](../frontend/app/page.tsx)
- [frontend/app/sku/[id]/page.tsx](../frontend/app/sku/[id]/page.tsx)
- [frontend/lib/api.ts](../frontend/lib/api.ts)
- [frontend/components/DataSourceBadge.tsx](../frontend/components/DataSourceBadge.tsx)

Goal: understand how the dashboard calls the backend and how provenance is
shown to the user.

### Final 5 minutes: Persistence and tests

Read:

- [backend/src/storage/analysis_store.py](../backend/src/storage/analysis_store.py)
- [backend/src/services/stock_service.py](../backend/src/services/stock_service.py)
- [backend/src/repositories/stock_repository.py](../backend/src/repositories/stock_repository.py)
- [backend/src/db/models.py](../backend/src/db/models.py)
- [backend/tests/](../backend/tests)

Goal: understand the old local analysis persistence, the newer service/repo
stock path, and how behavior is protected by tests.

## 4. What Happens When a User Clicks Analyze?

There are two common entry points:

- Dashboard loads many SKUs and runs `POST /api/analyze` for each row.
- SKU detail page runs `POST /api/analyze` when stock, lead time, or service
  level changes.

The flow is:

```text
Frontend
  -> api.analyzeSku(...)
  -> POST /api/analyze
  -> main.py validates AnalyzeRequest
  -> resolve demand history
  -> IntelligentInventoryService
  -> adaptive forecast routing
  -> reorder-point calculation
  -> business constraints
  -> response composition
  -> AnalysisStore snapshot
  -> frontend renders recommendation
```

Key files:

- Request schema: [backend/main.py](../backend/main.py)
- API client: [frontend/lib/api.ts](../frontend/lib/api.ts)
- Decision service: [backend/src/services/intelligent_inventory_service.py](../backend/src/services/intelligent_inventory_service.py)
- Forecast routing: [backend/src/services/adaptive_forecasting_service.py](../backend/src/services/adaptive_forecasting_service.py)
- Inventory math: [backend/src/inventory/reorder_point.py](../backend/src/inventory/reorder_point.py)

The response includes:

- risk bucket,
- forecast values,
- recommended order quantity,
- demand pattern,
- forecast method,
- demand provenance,
- forecast provenance,
- model/method metadata,
- explanation strings,
- decision math details.

## 5. Where Does the Data Come From?

The project uses UCI Online Retail II as its demo dataset.

```text
data/raw/online_retail_II.csv
  -> load_and_clean_retail_data()
  -> aggregate_daily_demand()
  -> data/processed/daily_demand.parquet
```

Important files:

- [backend/src/ingestion/load_retail_data.py](../backend/src/ingestion/load_retail_data.py)
- [backend/scripts/train_model.py](../backend/scripts/train_model.py)
- [backend/src/services/data_service.py](../backend/src/services/data_service.py)

`DataService` reads the processed parquet and serves demand history to API
routes. It is also used by SKU listing and history endpoints.

If a SKU is unknown, `/api/analyze` uses deterministic synthetic demand. That
fallback is deliberately labeled as `demand_source: "synthetic"` so the UI does
not pretend it is real sales data.

## 6. Where Does AI Run?

The ML path runs inside the backend during `/api/analyze`.

For regular-demand SKUs:

```text
demand history
  -> build_inference_features(...)
  -> trained LightGBM model
  -> recursive forecast_next_days(...)
```

Important files:

- [backend/src/services/adaptive_forecasting_service.py](../backend/src/services/adaptive_forecasting_service.py)
- [backend/src/features/inference_features.py](../backend/src/features/inference_features.py)
- [backend/src/forecasting/forecast_service.py](../backend/src/forecasting/forecast_service.py)
- [backend/src/services/model_service.py](../backend/src/services/model_service.py)

For intermittent or highly intermittent SKUs, the system intentionally avoids
LightGBM and uses statistical methods:

- `croston_forecast(...)` for intermittent demand,
- `conservative_forecast(...)` for highly intermittent demand.

This is not a weakness. The committed evaluation artifact shows LightGBM is not
dominant on sparse retail demand, especially intermittent SKUs. That is why the
project uses adaptive routing instead of forcing every SKU through one model.

## 7. Where Is the Inventory Logic?

Inventory logic lives under `backend/src/inventory/` and is called by
`IntelligentInventoryService`.

Core formula:

```text
lead_time_demand = sum(forecast over lead time)
safety_stock = buffer for uncertainty
reorder_point = lead_time_demand + safety_stock
order_quantity = max(0, reorder_point - current_stock)
```

Then business constraints can modify the raw order quantity:

- minimum order quantity,
- order multiple rounding,
- max order cap,
- budget-aware allocation across SKUs.

Important files:

- [backend/src/inventory/reorder_point.py](../backend/src/inventory/reorder_point.py)
- [backend/src/inventory/business_constraints.py](../backend/src/inventory/business_constraints.py)
- [backend/src/uncertainty/dynamic_sigma.py](../backend/src/uncertainty/dynamic_sigma.py)
- [backend/src/uncertainty/prediction_intervals.py](../backend/src/uncertainty/prediction_intervals.py)

## 8. How Does the Backend Work?

The backend is a FastAPI app in [backend/main.py](../backend/main.py). It owns:

- app startup/lifespan,
- model loading,
- data loading,
- auth dependencies,
- request/response schemas,
- API routes.

The backend currently has two architecture styles:

### Older analyze path

```text
HTTP route
  -> main.py orchestration
  -> IntelligentInventoryService
  -> AnalysisStore
```

This path works and is well tested, but `main.py` still owns too much response
composition. A future refactor should extract `AnalysisService`.

### Newer stock path

```text
HTTP route
  -> StockService
  -> StockRepository
  -> SQLAlchemy models
  -> database
```

This is the target layering for future persistence work.

Important files:

- [backend/src/services/stock_service.py](../backend/src/services/stock_service.py)
- [backend/src/dependencies/stock.py](../backend/src/dependencies/stock.py)
- [backend/src/repositories/stock_repository.py](../backend/src/repositories/stock_repository.py)
- [backend/src/db/session.py](../backend/src/db/session.py)
- [backend/src/db/models.py](../backend/src/db/models.py)

## 9. How Does the Frontend Work?

The frontend is a Next.js app.

Main pages:

- [frontend/app/page.tsx](../frontend/app/page.tsx): dashboard, KPIs, SKU table,
  recent analyses.
- [frontend/app/sku/[id]/page.tsx](../frontend/app/sku/[id]/page.tsx): detailed
  SKU analysis, history chart, stock input, lead time/service level controls.

Shared client/API code:

- [frontend/lib/api.ts](../frontend/lib/api.ts): typed backend client.
- [frontend/lib/stock.ts](../frontend/lib/stock.ts): browser-local fallback for
  stock values.
- [frontend/components/DataSourceBadge.tsx](../frontend/components/DataSourceBadge.tsx):
  provenance labels.
- [frontend/components/EmptyState.tsx](../frontend/components/EmptyState.tsx):
  honest missing-data states.

Frontend data flow:

```text
page loads
  -> fetch health, KPIs, SKUs, stock
  -> compute fallback demo stock when needed
  -> call /api/analyze
  -> render risk, method, order quantity, provenance
```

Server-side stock is preferred. Browser-local stock remains a fallback if the
stock endpoint is unavailable or no server-side stock has been recorded yet.

## 10. How Does the Database Fit In?

There are currently two persistence layers:

### Local/demo analysis persistence

[backend/src/storage/analysis_store.py](../backend/src/storage/analysis_store.py)
uses Python `sqlite3` to persist recent `/api/analyze` snapshots. This powers
the recent analyses panel and keeps the demo from feeling fully transient.

### SQLAlchemy stock and future production persistence

The newer database foundation is:

- [backend/src/db/models.py](../backend/src/db/models.py)
- [backend/src/db/session.py](../backend/src/db/session.py)
- [backend/src/repositories/stock_repository.py](../backend/src/repositories/stock_repository.py)
- [backend/alembic/versions/50f995297bfe_initial_inventory_schema.py](../backend/alembic/versions/50f995297bfe_initial_inventory_schema.py)
- [docs/database-design.md](./database-design.md)

The schema includes tables for:

- SKUs,
- stock levels,
- inventory policies,
- analysis runs,
- prediction logs,
- forecast evaluations,
- model artifacts.

Only server-side stock endpoints are wired into the new SQLAlchemy path today.
Analysis runs still use the older SQLite `AnalysisStore`. That split is
intentional during the migration, but it should not remain forever.

## 11. How Do I Debug It?

Start with `/health`.

```bash
curl http://localhost:8000/health
```

If `model_loaded` is false:

- Check `backend/saved_models/`.
- Run `python backend/scripts/bootstrap.py`.
- Check `MODEL_PATH`.

If `data_available` is false:

- Check `data/raw/online_retail_II.csv`.
- Check `data/processed/daily_demand.parquet`.
- Run `python backend/scripts/bootstrap.py`.

If KPIs are missing:

- Run `python backend/scripts/compute_kpis.py`.
- Check `backend/data/cached_kpis.json`.

If stock endpoints return 503:

- Run migrations:

```bash
cd backend
python -m alembic upgrade head
```

If frontend cannot connect:

- Check `NEXT_PUBLIC_API_URL`.
- Confirm backend is running on port `8000`.
- Confirm CORS `ALLOWED_ORIGINS` includes the frontend origin.

If recommendations look strange:

- Check `demand_source`: historical, request, or synthetic.
- Check `forecast_source`: model, statistical, or rule-based.
- Check `forecast_method`.
- Check current stock provenance in the UI.
- Inspect `backend/data/forecast_evaluation.json` before assuming LightGBM
  should win.

## 12. Which Files Should I Read First?

If you only read ten files, read these:

1. [README.md](../README.md)
2. [docs/architecture.md](./architecture.md)
3. [docs/api.md](./api.md)
4. [backend/main.py](../backend/main.py)
5. [backend/src/services/intelligent_inventory_service.py](../backend/src/services/intelligent_inventory_service.py)
6. [backend/src/services/adaptive_forecasting_service.py](../backend/src/services/adaptive_forecasting_service.py)
7. [backend/src/inventory/reorder_point.py](../backend/src/inventory/reorder_point.py)
8. [frontend/lib/api.ts](../frontend/lib/api.ts)
9. [frontend/app/sku/[id]/page.tsx](../frontend/app/sku/[id]/page.tsx)
10. [backend/tests/test_forecasting.py](../backend/tests/test_forecasting.py)

If you are reviewing backend architecture, also read:

- [backend/src/services/stock_service.py](../backend/src/services/stock_service.py)
- [backend/src/repositories/stock_repository.py](../backend/src/repositories/stock_repository.py)
- [backend/src/db/models.py](../backend/src/db/models.py)

If you are reviewing ML quality, also read:

- [backend/scripts/evaluate_forecast.py](../backend/scripts/evaluate_forecast.py)
- [backend/scripts/evaluate_cross_sku.py](../backend/scripts/evaluate_cross_sku.py)
- [backend/src/evaluation/metrics.py](../backend/src/evaluation/metrics.py)

## 13. What Is Real, Demo, and Still In Progress?

Real:

- Historical demand from the processed dataset.
- LightGBM inference when the artifact is loaded.
- Croston/conservative statistical forecasting.
- Reorder point and safety stock calculations.
- Business constraints.
- Forecast evaluation against baselines.
- Server-side stock endpoints after migrations.
- Backend tests around forecasting, inventory, APIs, settings, persistence, and
  stock services.

Demo or limited:

- The dataset is public retail data, not a live business feed.
- Unknown SKUs use deterministic synthetic demand.
- Recent analysis persistence still uses a local SQLite `AnalysisStore`.
- Authentication is demo-grade, not production identity.
- Frontend stock falls back to browser storage when the stock API has no value.

In progress:

- Full migration from local analysis SQLite to SQLAlchemy repositories.
- Prediction logging into `prediction_logs`.
- Evidence-based model routing from stored forecast evaluations.
- Extracting `/api/analyze` orchestration into `AnalysisService`.
- Auditing unused CopilotKit scaffolding.

## 14. What Should Be Improved Next?

The highest-value next PR is to extract an `AnalysisService` for
`POST /api/analyze` while preserving the exact API response shape.

Why:

- `main.py` currently owns too much orchestration.
- Prediction logging should not be added directly to the route.
- The newer stock flow already shows the preferred architecture:

```text
HTTP route
  -> Service
  -> Repository
  -> Database
```

The analyze path should move in that direction before more persistence or model
routing logic is added.

