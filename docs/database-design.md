# Database Design

This document defines the target persistence model for SupplySync AI before
introducing SQLAlchemy, Alembic, or PostgreSQL code.

The current repository has two persistence paths:

- `/api/analyze` snapshots are stored in a local SQLite table through
  `backend/src/storage/analysis_store.py`.
- editable stock levels are stored in browser `localStorage` through
  `frontend/lib/stock.ts`.

That is useful for a demo, but it is not enough for a production inventory
decision system. The target database must store inventory state, inventory
policies, model artifacts, evaluation results, analysis runs, and prediction
logs in normalized tables.

## Design Goals

- Preserve existing API response shapes while changing persistence underneath.
- Make stock levels server-side and auditable.
- Record every forecast decision so predictions can be compared to actual demand
  later.
- Keep model evaluation evidence queryable for routing decisions.
- Support PostgreSQL first, with SQLite usable for lightweight local tests where
  practical.
- Avoid multi-tenant complexity until the single-organization workflow is stable.

## Entity Overview

```
skus
  |
  | 1-to-many
  v
stock_levels

skus
  |
  | 1-to-many
  v
inventory_policies

skus
  |
  | 1-to-many
  v
analysis_runs
  |
  | 1-to-one / 1-to-many
  v
prediction_logs

model_artifacts
  |
  | 1-to-many
  v
forecast_evaluations

skus
  |
  | optional many evaluations per SKU
  v
forecast_evaluations
```

## Tables

### `skus`

Canonical product/SKU dimension table. Demand history may still be loaded from
the processed parquet during the first migration, but SKU metadata belongs in
the database.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `bigserial` | primary key | Internal database id. |
| `sku_code` | `varchar(64)` | unique, not null | External SKU identifier, e.g. `85099B`. |
| `name` | `text` | nullable | Product description from the source dataset or user import. |
| `is_active` | `boolean` | not null, default `true` | Allows hiding discontinued SKUs without deleting history. |
| `created_at` | `timestamptz` | not null, default `now()` | Insert timestamp. |
| `updated_at` | `timestamptz` | not null, default `now()` | Updated by application or trigger. |

Indexes:

- `uq_skus_sku_code` on `sku_code`
- `idx_skus_is_active` on `is_active`

Main queries:

- List active SKUs for `/api/skus`.
- Resolve `sku_code` to internal `sku_id` for analysis, stock, and prediction
  logging.
- Search SKUs by code or product name in the dashboard.

Relationships:

- One SKU has many stock level records.
- One SKU has many inventory policies.
- One SKU has many analysis runs.
- One SKU has many forecast evaluations.

### `stock_levels`

Current and historical stock snapshots. This replaces browser `localStorage` as
the source of truth for user-edited stock.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `bigserial` | primary key | Snapshot id. |
| `sku_id` | `bigint` | foreign key to `skus.id`, not null | SKU being updated. |
| `quantity_on_hand` | `numeric(14, 3)` | not null, check `>= 0` | Current physical stock estimate. |
| `quantity_reserved` | `numeric(14, 3)` | not null, default `0`, check `>= 0` | Reserved stock, optional for future workflows. |
| `quantity_available` | `numeric(14, 3)` | not null, check `>= 0` | Usually `on_hand - reserved`; stored for query speed and audit clarity. |
| `source` | `varchar(32)` | not null | `user`, `import`, `api`, or `demo_seed`. |
| `note` | `text` | nullable | Optional human note or import filename. |
| `recorded_at` | `timestamptz` | not null, default `now()` | Business timestamp for the stock value. |
| `created_at` | `timestamptz` | not null, default `now()` | Database insert timestamp. |

Indexes:

- `idx_stock_levels_sku_recorded_at` on `(sku_id, recorded_at desc)`
- `idx_stock_levels_recorded_at` on `recorded_at desc`
- Optional future partial index for latest stock materialization if a separate
  `current_stock_levels` table is not introduced.

Main queries:

- `GET /api/stock`: latest stock for all active SKUs.
- `GET /api/stock/{sku_id}`: latest stock for one SKU.
- `PUT /api/stock/{sku_id}`: append a new stock snapshot.
- Retrieve historical stock snapshots for audit or charts.

Relationships:

- Many stock snapshots belong to one SKU.
- Analysis runs should copy the stock value used at decision time so old
  decisions remain reproducible even if stock later changes.

Notes:

- Do not update old rows in place for normal stock changes. Append snapshots.
- A later optimization can add a `current_stock_levels` materialized table or
  view, but the first version can query latest-by-SKU safely with an index.

### `inventory_policies`

Stores configurable planning assumptions per SKU. This moves hardcoded/default
business policy into data while preserving app-level defaults.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `bigserial` | primary key | Policy id. |
| `sku_id` | `bigint` | foreign key to `skus.id`, not null | SKU this policy applies to. |
| `lead_time_days` | `integer` | not null, check `between 1 and 365` | Supplier lead time. |
| `service_level` | `numeric(5, 4)` | not null, check `> 0 and < 1` | Target service level, e.g. `0.9500`. |
| `moq` | `integer` | not null, default `1`, check `>= 0` | Minimum order quantity. |
| `order_multiple` | `integer` | not null, default `1`, check `>= 1` | Rounding multiple. |
| `max_order_quantity` | `integer` | nullable, check `> 0` | Supplier cap, if any. |
| `holding_cost_per_unit` | `numeric(12, 4)` | nullable, check `>= 0` | Per-unit holding cost override. |
| `stockout_cost_per_unit` | `numeric(12, 4)` | nullable, check `>= 0` | Per-unit stockout cost override. |
| `is_active` | `boolean` | not null, default `true` | Allows policy history. |
| `effective_from` | `timestamptz` | not null, default `now()` | Policy start. |
| `created_at` | `timestamptz` | not null, default `now()` | Insert timestamp. |
| `updated_at` | `timestamptz` | not null, default `now()` | Update timestamp. |

Indexes:

- `idx_inventory_policies_sku_active` on `(sku_id, is_active)`
- `idx_inventory_policies_effective_from` on `effective_from desc`
- Optional unique partial index: one active policy per SKU.

Main queries:

- Load planning defaults for `/api/analyze`.
- Update lead time, service level, MOQ, multiple, or max order per SKU.
- Audit policy changes that affected a previous recommendation.

Relationships:

- Many policies can belong to one SKU over time.
- Analysis runs should copy the policy values used at decision time.

### `analysis_runs`

Durable record of every inventory decision returned by `/api/analyze`. This
replaces the current single SQLite `analyses` table while preserving its
dashboard use case.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `bigserial` | primary key | Analysis id. |
| `request_id` | `uuid` | unique, not null | Stable id for tracing one API request. |
| `sku_id` | `bigint` | foreign key to `skus.id`, nullable | Null only for unknown/synthetic SKU requests. |
| `sku_code` | `varchar(64)` | not null | Copied request SKU for audit, even when unknown. |
| `current_stock` | `numeric(14, 3)` | not null, check `>= 0` | Stock used in the decision. |
| `recommended_order_quantity` | `integer` | not null, check `>= 0` | Final order quantity after constraints. |
| `action` | `varchar(32)` | not null | Existing API action, e.g. `PURCHASE` or no-action equivalent. |
| `risk` | `varchar(16)` | not null | `HIGH`, `MEDIUM`, or `LOW`. |
| `risk_color` | `varchar(16)` | nullable | Existing UI color, copied for compatibility. |
| `demand_pattern` | `varchar(32)` | not null | `regular`, `intermittent`, `highly_intermittent`. |
| `demand_source` | `varchar(32)` | not null | `historical`, `request`, or `synthetic`. |
| `forecast_source` | `varchar(32)` | not null | `model_forecast`, `statistical_method`, `rule_based_estimate`, `unavailable`. |
| `forecast_method` | `varchar(64)` | not null | `ml_lightgbm`, `croston`, `conservative`, `simple_average`, etc. |
| `routing_reason` | `text` | nullable | Human-readable model-routing reason. |
| `lead_time_days` | `integer` | not null | Copied assumption. |
| `service_level` | `numeric(5, 4)` | not null | Copied assumption. |
| `lead_time_demand` | `numeric(14, 3)` | nullable | Sum of forecast over lead time. |
| `safety_stock` | `numeric(14, 3)` | nullable | Computed safety buffer. |
| `safety_stock_method` | `varchar(32)` | nullable | `traditional` or `dynamic`. |
| `reorder_point` | `numeric(14, 3)` | nullable | Decision threshold. |
| `inventory_gap` | `numeric(14, 3)` | nullable | `max(0, reorder_point - current_stock)`. |
| `p50` | `numeric(14, 3)` | nullable | Existing response forecast p50. |
| `p90` | `numeric(14, 3)` | nullable | Existing response forecast p90. |
| `forecast_daily` | `jsonb` | nullable | Daily horizon output for audit/replay. |
| `explanation` | `jsonb` | nullable | Existing deterministic explanation block. |
| `created_at` | `timestamptz` | not null, default `now()` | API decision timestamp. |

Indexes:

- `idx_analysis_runs_created_at` on `created_at desc`
- `idx_analysis_runs_sku_created_at` on `(sku_id, created_at desc)`
- `idx_analysis_runs_sku_code_created_at` on `(sku_code, created_at desc)`
- `idx_analysis_runs_risk_created_at` on `(risk, created_at desc)`
- `uq_analysis_runs_request_id` on `request_id`

Main queries:

- `GET /api/analyses/recent?limit=N`.
- Recent decisions for a single SKU.
- Audit which stock, policy, and model path produced a recommendation.
- Join with prediction logs to compare forecast output with actual demand later.

Relationships:

- Many analysis runs may belong to one SKU.
- One analysis run should have at least one prediction log row for the current
  forecast path.

### `prediction_logs`

ML-system audit table. Every forecast call writes one row here so the system can
later compare predicted demand to actual observed demand.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `bigserial` | primary key | Prediction id. |
| `analysis_run_id` | `bigint` | foreign key to `analysis_runs.id`, nullable | Null only for non-API batch predictions. |
| `sku_id` | `bigint` | foreign key to `skus.id`, nullable | Null for unknown synthetic SKUs. |
| `sku_code` | `varchar(64)` | not null | Copied SKU code for audit. |
| `predicted_at` | `timestamptz` | not null, default `now()` | Forecast timestamp. |
| `target_start_date` | `date` | nullable | First date forecasted, if known. |
| `target_end_date` | `date` | nullable | Last date forecasted, if known. |
| `demand_source` | `varchar(32)` | not null | `historical`, `request`, or `synthetic`. |
| `forecast_method` | `varchar(64)` | not null | Method used. |
| `forecast_source` | `varchar(32)` | not null | Provenance category. |
| `routing_reason` | `text` | nullable | Why this model/method was selected. |
| `model_name` | `varchar(128)` | nullable | `lightgbm_demand_forecast`, `croston_sba`, etc. |
| `model_version` | `varchar(128)` | nullable | Artifact version/hash/timestamp. |
| `model_artifact_id` | `bigint` | foreign key to `model_artifacts.id`, nullable | Populated for trained artifacts. |
| `input_history_length` | `integer` | not null, check `>= 0` | Number of demand observations used. |
| `forecast_horizon_days` | `integer` | not null, check `> 0` | Number of future days predicted. |
| `p50` | `numeric(14, 3)` | nullable | Prediction summary. |
| `p90` | `numeric(14, 3)` | nullable | Prediction summary. |
| `forecast_daily` | `jsonb` | nullable | Daily values. |
| `recommended_order_quantity` | `integer` | not null, check `>= 0` | Recommendation linked to this forecast. |
| `actual_observed_demand` | `numeric(14, 3)` | nullable, check `>= 0` | Filled later when actuals are known. |
| `actual_observed_at` | `timestamptz` | nullable | When actual demand was attached. |
| `created_at` | `timestamptz` | not null, default `now()` | Insert timestamp. |

Indexes:

- `idx_prediction_logs_predicted_at` on `predicted_at desc`
- `idx_prediction_logs_sku_predicted_at` on `(sku_id, predicted_at desc)`
- `idx_prediction_logs_method_predicted_at` on `(forecast_method, predicted_at desc)`
- `idx_prediction_logs_actual_missing` partial index on `(target_end_date)` where
  `actual_observed_demand is null`
- `idx_prediction_logs_model_artifact` on `model_artifact_id`

Main queries:

- Log every `/api/analyze` forecast.
- Find predictions whose actual demand can now be filled.
- Compute live monitoring metrics: MAE, bias, WAPE, forecast drift by SKU,
  method, and model version.
- Explain to a recruiter how offline evaluation becomes online monitoring.

Relationships:

- Prediction logs optionally belong to an analysis run.
- Prediction logs optionally reference a model artifact.
- Prediction logs optionally belong to a known SKU.

### `forecast_evaluations`

Stores offline evaluation evidence used by model routing. Existing JSON output
from `backend/data/forecast_evaluation.json` can be imported here later.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `bigserial` | primary key | Evaluation row id. |
| `model_artifact_id` | `bigint` | foreign key to `model_artifacts.id`, nullable | Null for non-artifact baselines. |
| `sku_id` | `bigint` | foreign key to `skus.id`, nullable | Null when row is class-level or global aggregate. |
| `sku_code` | `varchar(64)` | nullable | Copied SKU code for import/debugging. |
| `demand_class` | `varchar(32)` | nullable | `regular`, `intermittent`, `highly_intermittent`, or `all`. |
| `model_name` | `varchar(128)` | not null | `lightgbm`, `croston_sba`, `moving_avg_7`, etc. |
| `evaluation_scope` | `varchar(32)` | not null | `global`, `demand_class`, `sku`, or `cross_sku_fold`. |
| `metric_mae` | `numeric(14, 6)` | nullable | Lower is better. |
| `metric_rmse` | `numeric(14, 6)` | nullable | Lower is better. |
| `metric_bias` | `numeric(14, 6)` | nullable | Signed forecast bias. |
| `metric_wape` | `numeric(14, 6)` | nullable | Lower is better. |
| `metric_mase` | `numeric(14, 6)` | nullable | Lower is better. |
| `n_skus` | `integer` | nullable | Aggregate count. |
| `n_test_points` | `integer` | nullable | Evaluation sample size. |
| `horizon_days` | `integer` | nullable | Evaluation horizon. |
| `source_path` | `text` | nullable | JSON/CSV artifact imported from. |
| `generated_at` | `timestamptz` | nullable | Timestamp from evaluation artifact. |
| `created_at` | `timestamptz` | not null, default `now()` | Insert timestamp. |

Indexes:

- `idx_forecast_eval_scope_class` on `(evaluation_scope, demand_class)`
- `idx_forecast_eval_sku` on `(sku_id, model_name)`
- `idx_forecast_eval_model` on `(model_name)`
- `idx_forecast_eval_generated_at` on `generated_at desc`

Main queries:

- Pick the best method for a demand class by `metric_mase`, falling back to
  `metric_wape` or `metric_mae`.
- Pick the best method for a specific SKU if per-SKU evidence exists.
- Show evaluation evidence in `/api/model-info`.
- Explain why LightGBM was or was not selected.

Relationships:

- Many evaluation rows can reference one model artifact.
- Evaluation rows may be global, demand-class-level, or SKU-level.

### `model_artifacts`

Registry table for trained model artifacts and statistical method versions.
This is not a full model registry; it is enough to make routing and prediction
logs reproducible.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `bigserial` | primary key | Artifact id. |
| `model_name` | `varchar(128)` | not null | `lightgbm_demand_forecast`, `croston_sba`, etc. |
| `model_type` | `varchar(64)` | not null | `ml`, `statistical_method`, `rule_based_fallback`. |
| `version` | `varchar(128)` | not null | Timestamp, semantic version, or artifact hash. |
| `artifact_uri` | `text` | nullable | Local path, object-store URI, or null for pure statistical methods. |
| `metadata_uri` | `text` | nullable | Path to metadata JSON, if separate. |
| `feature_schema` | `jsonb` | nullable | Ordered feature names for ML artifacts. |
| `training_dataset` | `text` | nullable | Dataset identifier. |
| `training_started_at` | `timestamptz` | nullable | Training start. |
| `training_finished_at` | `timestamptz` | nullable | Training end. |
| `training_metrics` | `jsonb` | nullable | Training MAE/RMSE and future metrics. |
| `is_active` | `boolean` | not null, default `false` | Runtime default candidate. |
| `created_at` | `timestamptz` | not null, default `now()` | Insert timestamp. |

Indexes:

- `uq_model_artifacts_name_version` on `(model_name, version)`
- `idx_model_artifacts_active` on `(model_name, is_active)`
- `idx_model_artifacts_type` on `model_type`

Main queries:

- Load active model metadata for `/api/model-info`.
- Attach `model_artifact_id` and `model_version` to prediction logs.
- Compare evaluation rows across artifact versions.

Relationships:

- One model artifact has many forecast evaluations.
- One model artifact can be referenced by many prediction logs.

## Main Runtime Queries

### Latest stock for all SKUs

```sql
select distinct on (s.id)
       s.sku_code,
       s.name,
       sl.quantity_on_hand,
       sl.quantity_available,
       sl.source,
       sl.recorded_at
from skus s
left join stock_levels sl on sl.sku_id = s.id
where s.is_active = true
order by s.id, sl.recorded_at desc;
```

Serves `GET /api/stock` and dashboard initialization.

### Latest stock for one SKU

```sql
select s.sku_code,
       sl.quantity_on_hand,
       sl.quantity_available,
       sl.source,
       sl.recorded_at
from skus s
left join stock_levels sl on sl.sku_id = s.id
where s.sku_code = :sku_code
order by sl.recorded_at desc
limit 1;
```

Serves `GET /api/stock/{sku_id}` and `/api/analyze` default stock lookup.

### Append stock update

```sql
insert into stock_levels (
    sku_id,
    quantity_on_hand,
    quantity_reserved,
    quantity_available,
    source,
    note
) values (
    :sku_id,
    :quantity_on_hand,
    :quantity_reserved,
    :quantity_available,
    :source,
    :note
);
```

Serves `PUT /api/stock/{sku_id}`.

### Recent analyses

```sql
select *
from analysis_runs
order by created_at desc
limit :limit;
```

Serves the existing recent analyses panel.

### Best routing method for a demand class

```sql
select model_name,
       metric_mase,
       metric_wape,
       metric_mae,
       generated_at
from forecast_evaluations
where evaluation_scope = 'demand_class'
  and demand_class = :demand_class
order by metric_mase asc nulls last,
         metric_wape asc nulls last,
         metric_mae asc nulls last,
         generated_at desc
limit 1;
```

Serves evidence-based model routing.

### Predictions waiting for actual demand

```sql
select *
from prediction_logs
where actual_observed_demand is null
  and target_end_date < current_date
order by target_end_date asc;
```

Serves future monitoring/backfill jobs.

## Migration Plan

### Phase 1: Design only

- Add this document.
- Do not add SQLAlchemy, Alembic, or PostgreSQL dependencies yet.

### Phase 2: ORM foundation

- Add SQLAlchemy, Alembic, and PostgreSQL driver dependencies.
- Create `src/db/session.py` for engine/session construction.
- Create `src/db/models.py` for table mappings.
- Create repositories for analysis and stock persistence.
- Keep `main.py` free of direct SQLAlchemy calls.

### Phase 3: Analysis persistence migration

- Implement `AnalysisRepository`.
- Preserve `GET /api/analyses/recent` response shape.
- Write to `analysis_runs` instead of the old SQLite `analyses` table.
- Keep the current `AnalysisStore` available only as a compatibility fallback
  until the new repository is tested.

### Phase 4: Server-side stock

- Implement `StockRepository`.
- Add `GET /api/stock`, `GET /api/stock/{sku_id}`, and
  `PUT /api/stock/{sku_id}`.
- Update the frontend to use the backend stock API.
- Keep a short-lived localStorage fallback only for offline/error states if
  needed, clearly labeled as degraded mode.

### Phase 5: Prediction logging

- Write a `prediction_logs` row for every `/api/analyze` call.
- Include model name, model version, demand source, forecast method, input
  history length, horizon, p50, p90, daily forecast, and recommended quantity.
- Leave `actual_observed_demand` nullable for later backfills.

### Phase 6: Evidence-based routing

- Import `forecast_evaluation.json` into `forecast_evaluations`.
- Prefer per-SKU evaluation winners when available.
- Fall back to demand-class winners.
- Fall back to threshold routing only when no evaluation evidence exists.
- Add `routing_reason` to the analyze response without removing existing fields.

## Transaction Boundaries

`PUT /api/stock/{sku_id}`:

- Resolve or create SKU.
- Insert stock snapshot.
- Commit.

`POST /api/analyze`:

- Resolve SKU.
- Load latest stock if request does not provide stock in a future API version.
- Load active inventory policy.
- Compute forecast and reorder decision outside the database transaction.
- Open transaction.
- Insert `analysis_runs`.
- Insert `prediction_logs`.
- Commit both rows together.

This keeps forecast computation from holding database locks while still making
the audit rows atomic.

## Deferred Tables

These are intentionally deferred until the single-organization flow is working:

- `users`
- `organizations`
- `organization_memberships`
- `suppliers`
- `purchase_orders`
- `stock_movements`
- `data_imports`

Adding them too early would make the schema look more enterprise-like without
improving the current product workflow.

