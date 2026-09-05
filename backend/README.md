# SupplySync AI — backend

FastAPI service that turns raw retail sales into inventory decisions. Adaptive forecasting (LightGBM / Croston / conservative buffer), Z-score safety stock, MOQ/order-multiple constraints, transparent provenance on every value.

For the product pitch, quickstart, and architecture diagrams, see the **[root README](../README.md)** and **[docs/](../docs/)**. This file is the orientation for engineers reading the backend in isolation.

---

## Layout

```
backend/
├── main.py                     # FastAPI app — all routes in one file
├── requirements.txt
├── .env.example                # All env vars documented here
├── Dockerfile
├── src/
│   ├── auth/                   # HMAC session cookie + optional API-key gate
│   ├── evaluation/             # MAE / RMSE / WAPE / MASE + baselines
│   ├── features/               # Feature row builder for LightGBM inference
│   ├── forecasting/            # LightGBM wrapper, Croston, moving-average fallback
│   ├── ingestion/              # UCI CSV → cleaned daily-demand parquet
│   ├── inventory/              # Safety stock, reorder point, order-qty constraints
│   ├── services/               # AdaptiveForecastingService (the method router)
│   ├── simulation/             # Day-by-day naive-vs-intelligent policy comparison
│   └── uncertainty/            # Rolling forecast error → dynamic safety stock
├── scripts/
│   ├── check_setup.py          # Prints [OK] / [MISSING] per required artifact
│   ├── bootstrap.py            # One-shot: trains + computes KPIs if missing
│   ├── train_model.py          # Fits LightGBM on top-20 SKUs; writes .pkl + metadata
│   ├── compute_kpis.py         # Naive vs intelligent cost simulation → cached KPIs
│   ├── evaluate_forecast.py    # LightGBM vs 4 baselines on temporal split
│   ├── evaluate_cross_sku.py   # NEW — unseen-SKU k-fold generalization test
│   └── evaluate_custom_dataset.py  # NEW — zero-shot evaluation on any compatible CSV
├── saved_models/               # lightgbm_demand_forecast.pkl + _metadata.json
├── data/                       # cached_kpis.json, forecast_evaluation.{json,csv}, historical_monitoring_replay.json (local dev DB: data/supplysync.db)
└── tests/                      # 280+ pytest cases
```

---

## Running locally

Requires Python 3.11+. From repo root:

```bash
make install        # installs backend + frontend deps
make bootstrap      # trains model, computes KPIs, fills backend/data/
make backend        # uvicorn on :8000 with --reload
```

Or directly:

```bash
cd backend
pip install -r requirements.txt
python scripts/bootstrap.py
uvicorn main:app --reload --port 8000
```

Health check: `curl http://localhost:8000/health` — every flag should be `true` when the model is loaded and KPIs are computed.

---

## Retraining

```bash
cd backend
python scripts/train_model.py
```

Writes `saved_models/lightgbm_demand_forecast.pkl` + `_metadata.json`. The
metadata includes version, SHA-256 checksum, feature-schema version, training
data summary, and lifecycle status. The live API validates those fields on
startup before loading the artifact.

Register and promote are explicit:

```bash
python scripts/register_model_artifact.py --model-name lightgbm_demand_forecast
python scripts/promote_model.py --artifact-id <id>
```

To train on a **different dataset**, set `DATA_CSV_PATH`:

```bash
DATA_CSV_PATH=/path/to/my_sales.csv python scripts/train_model.py
```

For non-UCI column names, pass a column mapping JSON. See **[docs/bring-your-own-data.md](../docs/bring-your-own-data.md)** for the full walkthrough.

---

## Evaluation

Three evaluation entry points, each a standalone script:

| Script | Answers |
|---|---|
| `evaluate_forecast.py` | "On the trained SKUs, does LightGBM beat the classical baselines on a temporal holdout?" |
| `evaluate_cross_sku.py` | "If we hold out SKUs the model never saw, how does it compare to baselines?" |
| `evaluate_custom_dataset.py` | "If I apply the current model to a different retail CSV, what do baselines vs LightGBM look like?" |

All three write JSON into `backend/data/` and print a readable summary to stdout. The cross-SKU script is the honest generalization proof — see its output in `backend/data/cross_sku_evaluation.json`.

---

## Tests

```bash
make test                          # or: cd backend && python -m pytest tests/ -q
```

Test files are organized by responsibility — `test_inventory_logic.py` for the reorder math, `test_forecasting.py` for method routing, `test_api_*.py` for the FastAPI surface, `test_cross_sku_generalization.py` for unseen-SKU and column-mapping behavior, etc.

---

## Environment variables

All documented in **[`.env.example`](./.env.example)** and composed through
`src/config/settings.py`, the single runtime settings entry point for the
FastAPI app. Individual domains live under `src/config/` (`app.py`, `auth.py`,
`database.py`, `forecasting.py`, `inventory.py`, `logging.py`) so new settings
do not turn into one giant module. Invalid numeric settings fail fast during
startup instead of silently falling back. Copy to `.env` or export in your
shell. Key ones:

- `MODEL_PATH` — where ModelService loads/saves the LightGBM artifact.
- `AUTH_MODE` — `off` (default) or `demo` (adds an HMAC session cookie gate).
- `ALLOWED_ORIGINS` — comma-separated CORS origins.
- `DATABASE_URL` — SQLAlchemy database URL. Docker/production should use PostgreSQL.
- `DATA_CSV_PATH` / `DATA_PARQUET_PATH` — override the default UCI paths for bring-your-own-data workflows.

---

## Where to look first

- **Routes** → [`main.py`](./main.py). Every endpoint is in one file by design — fast to scan.
- **Method routing** (regular → LightGBM, intermittent → Croston, highly intermittent → conservative buffer) → [`src/services/adaptive_forecasting_service.py`](./src/services/adaptive_forecasting_service.py).
- **Reorder math** → [`src/inventory/`](./src/inventory/).
- **How the API response is assembled** (risk, decision block, explanation, model_info) → [`src/services/analysis_service.py`](./src/services/analysis_service.py).
- **Persistence** → [`src/repositories/analysis_repository.py`](./src/repositories/analysis_repository.py), [`src/repositories/stock_repository.py`](./src/repositories/stock_repository.py), and [`src/db/session.py`](./src/db/session.py).

---

## Database migration foundation

The production database layer is now route-wired for stock, analysis history,
and prediction logging. Runtime persistence uses SQLAlchemy sessions,
repositories, and Alembic-managed tables.

- `DATABASE_URL` is the SQLAlchemy URL for the target PostgreSQL-backed persistence layer.
- `src/db/session.py` builds SQLAlchemy engines and sessions from `DATABASE_URL`.
- `src/db/models.py` maps the target schema from `docs/database-design.md`.
- `src/repositories/analysis_repository.py` owns analysis-run and prediction-log writes.
- `src/repositories/stock_repository.py` owns server-side stock snapshots.
- `alembic/` is configured to autogenerate migrations from `db.models.Base.metadata`.

Run migrations from `backend/`:

```bash
python -m alembic upgrade head
```

Route handlers should continue to avoid direct SQLAlchemy calls. They receive
services through dependencies; services depend on repositories; repositories
own SQLAlchemy details.

---

## Design notes

- **Honest fallbacks.** If the LightGBM artifact isn't loadable or the feature row can't be built, the API falls back to a 7-day moving average and labels `forecast_source` as `rule_based_estimate` — it never silently degrades.
- **Deterministic synthetic data.** Unknown SKUs get a Poisson(20)×30 series seeded on the SKU string, labeled `demand_source: "synthetic"`. No hidden randomness between calls.
- **No LLM in the response path.** Explanations are template-generated from the decision block — every sentence is traceable to a number the system already computed.
- **Persistence is required for analysis auditability.** If SQLAlchemy
  persistence is unavailable, analysis history endpoints return 503 instead of
  silently falling back to an untracked local store.
