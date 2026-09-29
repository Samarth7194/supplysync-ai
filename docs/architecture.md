# Architecture

A one-page technical tour of how SupplySync actually works at runtime. See the root [README](../README.md) for product context and evaluation numbers, and [docs/api.md](./api.md) for endpoint examples.

---

## Overall shape

```
           ┌─────────────────────────────────────────────────────────────┐
           │                    data/raw/online_retail_II.csv             │
           │                           (UCI dataset, ~1M rows)            │
           └──────────────────────────────┬──────────────────────────────┘
                                          │  scripts/train_model.py (1×)
                                          ▼
           ┌─────────────────────────────────────────────────────────────┐
           │             data/processed/daily_demand.parquet              │
           │              backend/saved_models/*.pkl + metadata.json      │
           │              backend/data/cached_kpis.json                   │
           │              backend/data/forecast_evaluation.{json,csv}     │
           └──────────────────────────────┬──────────────────────────────┘
                                          │  uvicorn main:app
                                          ▼
  ┌────────────────────────┐   POST /api/analyze   ┌──────────────────────┐
  │  Next.js 16 dashboard  │◀─────────────────────▶│  FastAPI backend     │
  │  app/page.tsx          │     GET /api/...      │  backend/main.py     │
  │  app/sku/[id]/page.tsx │                       │  services/ + inventory/
  └────────────────────────┘                       └──────────────────────┘
```

Everything below the cut describes what lives at each arrow.

---

## 1. Data preparation flow

Where: [`backend/src/ingestion/load_retail_data.py`](../backend/src/ingestion/load_retail_data.py), [`backend/scripts/train_model.py`](../backend/scripts/train_model.py).

1. `load_and_clean_retail_data()` reads the raw CSV, drops cancellations / invalid prices / service codes, and keeps only alphanumeric stock codes.
2. `aggregate_daily_demand()` groups by (StockCode, date) and writes `data/processed/daily_demand.parquet`.
3. `get_top_skus()` picks SKUs with at least 60 days of history.

The parquet is the single source of truth for everything downstream — training, live inference, the history endpoint, and the KPI simulator all read it.

## 2. Feature engineering

Where: [`backend/src/features/lag_features.py`](../backend/src/features/lag_features.py), [`time_features.py`](../backend/src/features/time_features.py), [`inference_features.py`](../backend/src/features/inference_features.py).

Training and live inference share the same primitives:

```
create_lag_features   → lag_1..lag_7, rolling_mean_7, rolling_std_7, rolling_mean_14
create_time_features  → day_of_week, month, is_weekend, day_of_month, week_of_year
```

At training time, features are precomputed over the entire history per SKU; at inference time, `build_inference_features(demand_series, feature_columns)` constructs a single row for "tomorrow" using the same two helpers so the model sees identical preprocessing.

## 3. Training / evaluation flow

Where: [`backend/scripts/train_model.py`](../backend/scripts/train_model.py), [`backend/scripts/evaluate_forecast.py`](../backend/scripts/evaluate_forecast.py), [`backend/src/evaluation/`](../backend/src/evaluation).

```
train_model.py                           evaluate_forecast.py
    │                                         │
    ├ load parquet                            ├ load parquet + trained artifact(s)
    ├ build per-SKU features                  ├ src/evaluation/backtest.py:
    ├ temporal split (last 30 days → test)    │   rolling-origin, multi-step —
    ├ fit LGBMRegressor                       │   at each origin, forecast the
    │   (--objective regression|tweedie|      │   whole horizon at once (no
    │    poisson)                             │   actuals fed back mid-horizon),
    ├ evaluate MAE / RMSE                     │   score vs what really followed
    ├ save pkl + metadata.json                ├ compare every method vs 4
    └ cache per-SKU last-feature-rows         │   reference baselines (predict_zero,
                                              │   moving_avg_7, seasonal_naive_7,
                                              │   croston_sba) — always included
                                              │   so nothing is judged in isolation
                                              ├ metrics: MAE / RMSE / bias / WAPE
                                              │   (daily + lead-time-sum) / MASE
                                              ├ aggregate per demand class
                                              └ write forecast_evaluation.{json,csv}
```

The same `backtest.py` module is the single evaluation engine for
`evaluate_forecast.py`, `candidate_evaluation_service.py`,
`historical_monitoring_replay_service.py`, `model_monitoring_service.py`, and
`model_routing_service.py` — one methodology, not several copy-pasted
one-step loops (that was a real bug, fixed; see the README's
[What I Found and Changed](../README.md#what-i-found-and-changed)).

Model metadata captures the feature schema (versioned — `demand_lag_calendar_v1`
or the newer `demand_lag_calendar_sku_v2` with added SKU-profile features),
train-SKU list, training MAE/RMSE, ISO-timestamped `saved_at`, artifact
checksum, explicit version, feature-schema version, and lifecycle status. The
evaluation artifact captures the generator timestamp, per-SKU metrics, and
aggregates with the same vocabulary the UI uses.

## 4. Live analyze flow

Where: [`backend/main.py`](../backend/main.py), [`backend/src/services/`](../backend/src/services).

For every `POST /api/analyze` request:

```
 body = { sku, current_stock, demand_history? }
    │
    ▼
 1. Resolve the demand series
    • request.demand_history           → demand_source = "request"
    • DataService.get_demand_history() → demand_source = "historical"
    • deterministic Poisson(20)×30     → demand_source = "synthetic"
    │
    ▼
 2. IntelligentInventoryService.get_intelligent_reorder_decision()
    ├ classify_sku_demand_pattern     (regular | intermittent | highly_intermittent)
    ├ ModelRoutingService.select_method   (evidence-based; on by default)
    │    reads the offline backtest evidence and only switches away from the
    │    legacy per-pattern default if a method beats every reference
    │    baseline there. With the evidence currently committed, this sends
    │    regular-demand SKUs to Croston-SBA instead of LightGBM.
    ├ adaptive_forecast               (runs whichever method was selected)
    │    ├ ml_lightgbm    → recursive_forecast: rebuilds lag/rolling/calendar
    │    │                  features from the extended series at *each* of
    │    │                  the H recursive steps (not one static row); if
    │    │                  any step fails, fall back to 7-day moving
    │    │                  average, logged with reason
    │    ├ croston        → croston_forecast (SBA bias correction)
    │    └ conservative   → conservative buffer (recent mean × 1.5)
    ├ compute dynamic / traditional safety stock
    ├ compute_reorder_decision        (reorder point, order quantity)
    └ apply_business_constraints      (MOQ, order multiple, max order)
    │
    ▼
 3. Compose response
    • forecast.{historical_mean_60d, historical_p90_60d, daily}
      (p50/p90 kept as deprecated aliases with identical values)
    • decision.{lead_time_days, lead_time_demand, safety_stock,
                reorder_point, service_level, inventory_gap, why}
      risk is HIGH if current_stock < lead_time_demand, MEDIUM if
      < reorder_point, else LOW — LOW never produces a nonzero order
    • routing.{selected_method, default_method, selection_source, reason, ...}
    • model_info.{model_name, model_type, artifact_available, ...}
    • demand_source / forecast_method / forecast_source
```

Every provenance field is truthful: if the ML model isn't loaded (`_loaded_model is None`), `model_info.artifact_available` is `false`, `forecast_method` is `simple_average`, and `forecast_source` is `rule_based_estimate` — all three consistently.

## 5. KPI / simulation flow

Where: [`backend/scripts/compute_kpis.py`](../backend/scripts/compute_kpis.py), [`backend/src/simulation/`](../backend/src/simulation).

`compute_kpis.py` runs a day-by-day simulation on the top 50 SKUs (a 14-day
warm-up excluded from every metric, 90 measured days) comparing three
policies, all deciding off inventory *position* (on-hand + on-order):

- **naive policy**: reorder 2 weeks of average demand when stock drops below 1 week of average demand.
- **moving_average_rop**: an (s, S) reorder-point policy with a fixed safety stock (computed once from the warm-up window) and a rolling demand-rate estimate.
- **intelligent policy**: the same `IntelligentInventoryService` the live API uses — the real loaded LightGBM model and evidence-based routing, not `model=None`.

It writes aggregate totals (holding cost, stockout cost, fill rate, cost
savings vs. whichever baseline was actually cheaper, a 95% CI across SKUs,
and sensitivity to the stockout:holding cost ratio) to
`backend/data/cached_kpis.json`. `GET /api/kpis` returns that file with an
`interpretation` block describing both baselines, assumptions (lead time = 7,
service level = 0.95, holding cost = 0.5/unit, stockout cost = 5/unit, 90-day
measured window, 14-day warm-up), and a one-sentence meaning per KPI. **The
current honest result is negative**: the intelligent policy costs more than
the naive baseline at these cost assumptions — see the README's
[What I Found and Changed](../README.md#what-i-found-and-changed).

## 6. Frontend consumption

Where: [`frontend/app/`](../frontend/app), [`frontend/lib/api.ts`](../frontend/lib/api.ts), [`frontend/components/`](../frontend/components).

```
/ (Dashboard)                             /sku/[id] (Detail)
  ├ GET /health                             ├ GET /api/skus/details   (name lookup)
  ├ GET /api/kpis                           ├ GET /api/skus/{sku}/history
  ├ GET /api/skus/details                   └ POST /api/analyze
  └ POST /api/analyze × N  (batched 5 at a time)
```

Every surface that displays a value renders its provenance via the shared [`DataSourceBadge`](../frontend/components/DataSourceBadge.tsx) vocabulary (`recorded` / `user input` / `synthetic demo` / `model forecast` / `statistical` / `rule-based` / `demo value` / `unavailable`). Missing-data states use the shared [`EmptyState`](../frontend/components/EmptyState.tsx) so a reviewer sees the same tone whether KPIs are uncomputed, history is absent, or the SKU is unknown.

## 7. Where the technical depth lives

| Area | Code of interest |
|---|---|
| ML inference wiring | [`adaptive_forecasting_service.py`](../backend/src/services/adaptive_forecasting_service.py), [`inference_features.py`](../backend/src/features/inference_features.py) |
| Inventory math | [`reorder_point.py`](../backend/src/inventory/reorder_point.py), [`business_constraints.py`](../backend/src/inventory/business_constraints.py) |
| Uncertainty | [`dynamic_sigma.py`](../backend/src/uncertainty/dynamic_sigma.py), [`prediction_intervals.py`](../backend/src/uncertainty/prediction_intervals.py) |
| Evaluation | [`evaluation/backtest.py`](../backend/src/evaluation/backtest.py) (shared rolling-origin engine), [`evaluation/metrics.py`](../backend/src/evaluation/metrics.py), [`evaluation/baselines.py`](../backend/src/evaluation/baselines.py), [`scripts/evaluate_forecast.py`](../backend/scripts/evaluate_forecast.py) |
| Simulation | [`inventory_simulator.py`](../backend/src/simulation/inventory_simulator.py), [`enhanced_simulator.py`](../backend/src/simulation/enhanced_simulator.py) |
| Decision composition | [`analysis_service.py`](../backend/src/services/analysis_service.py), [`intelligent_inventory_service.py`](../backend/src/services/intelligent_inventory_service.py) |
| Provenance vocabulary | `classify_forecast_source`, `_model_info_for_method` in [`analysis_service.py`](../backend/src/services/analysis_service.py) + [`DataSourceBadge.tsx`](../frontend/components/DataSourceBadge.tsx) |

## 8. What is real vs demo

- **Real**: demand history from the parquet (fed to analyze when the SKU is known), the trained LightGBM predictions when the pkl is loaded, Croston / conservative forecasts, simulated KPIs, evaluation numbers.
- **Demo**: `current_stock` shown on the dashboard and SKU page (seeded from an approximation of each SKU's own reorder point — lead-time demand + safety stock from its average demand and demand std — hashed deterministically per SKU so some SKUs land above their reorder point and correctly show no action needed; clearly labeled with a `DEMO` pill). A real integration would pass live stock as the `current_stock` request field.
- **Synthetic fallback**: Poisson(20) demand when the requested SKU isn't in the processed dataset — flagged with `demand_source: "synthetic"` and an amber page-top banner.

## 9. Analysis Service Boundary

The analyze path has been moved toward the same service/repository layering as
the stock path:

```text
FastAPI route
  -> AnalysisService
  -> DataService / IntelligentInventoryService
  -> AnalysisRepository
  -> SQLAlchemy
```

`backend/main.py` still defines the HTTP route, request model, and response
model for backward compatibility. The orchestration now lives in
`backend/src/services/analysis_service.py`, which owns:

- demand-history resolution,
- request-provided versus historical versus synthetic demand provenance,
- invocation of `IntelligentInventoryService`,
- risk bucket calculation,
- response data composition,
- deterministic explanation text,
- SQLAlchemy analysis persistence,
- linked prediction-log creation.

`GET /api/analyses/recent` uses the same service and reads SQLAlchemy
`analysis_runs` rows. If persistence is unavailable, the API returns a 503
instead of silently writing to a second local store.

The SQLAlchemy write path stores two rows together through
`AnalysisRepository`:

```text
analysis_runs
  -> prediction_logs
```

The repository does not commit directly. The FastAPI SQLAlchemy session
dependency owns commit/rollback, which keeps transaction ownership in one
place.

## 10. Logged Prediction Evaluation

Logged prediction evaluation is separate from offline model benchmarking.

```text
POST /api/analyze
  -> analysis_runs
  -> prediction_logs
  -> forecast horizon completes
  -> DataService recorded actual demand
  -> ForecastEvaluationService
  -> forecast_evaluations
```

`prediction_logs` records the forecast method, source, horizon, target window,
model name/version, optional `sku_id`, optional `model_artifact_id`, forecast
values, and recommended quantity. SQLAlchemy links `analysis_runs` to the
prediction log that came from the request.

`ForecastEvaluationService` evaluates only predictions whose target window has
completed and whose actual demand window is available from `DataService`.
It does not evaluate synthetic/request-only predictions unless a valid recorded
actual window exists for the SKU. It never invents actuals.

Metric computation reuses `backend/src/evaluation/metrics.py`: MAE, RMSE,
bias, WAPE, and MASE. WAPE remains nullable when actual demand sums to zero,
which avoids MAPE-style failures on intermittent demand.

Offline files such as `backend/data/forecast_evaluation.json` remain model
benchmark evidence. Rows with `evaluation_scope="logged_prediction"` are
post-hoc evaluations of individual persisted predictions.

## 11. Model Artifact Lifecycle

The LightGBM model path is versioned and validated:

```text
train_model.py
  -> lightgbm_demand_forecast.pkl
  -> metadata JSON with checksum/version/feature schema
  -> register_model_artifact.py
  -> model_artifacts(status=candidate)
  -> evaluate
  -> promote_model.py
  -> model_artifacts(status=active)
  -> ModelService validates and loads
  -> prediction_logs keep exact model_artifact_id
```

Croston, conservative, and simple-average methods remain deterministic
statistical/rule-based methods. They carry method versions in prediction logs
but do not get fake model artifact rows.

## 12. Evidence-Based Routing

Routing remains demand-pattern aware. `ModelRoutingService` decides; the
adaptive forecasting service executes.

```text
Demand History
  -> Pattern Classifier
  -> Eligible Methods
  -> ModelRoutingService
       -> insufficient evidence: default policy
       -> sufficient evidence: metric comparison
  -> Selected Method
  -> AdaptiveForecastingService
  -> Forecast
  -> Prediction Log
  -> Forecast Evaluation
  -> Future routing evidence
```

Default policy is unchanged:

- `regular -> ml_lightgbm`, with existing `simple_average` fallback if the
  artifact/features are unavailable.
- `intermittent -> croston`.
- `highly_intermittent -> conservative`.

Eligible methods are intentionally narrow. Regular SKUs may compare
`ml_lightgbm`, `croston`, and logged `simple_average` evidence. Intermittent
SKUs may compare `croston` and logged `simple_average` evidence.
Highly-intermittent SKUs keep `conservative` until there is real evidence for a
safe alternative.

Evidence hierarchy:

1. SKU-level logged evaluations.
2. Demand-pattern logged evaluations.
3. SKU-level offline benchmark evidence.
4. Demand-pattern offline benchmark evidence.
5. Default routing.

The primary routing metric is WAPE. Evidence must be recent, horizon-matching,
sample-size sufficient, and materially better than the default method. Ties and
small improvements keep the default method to avoid method flapping.

## 13. MLOps: Monitoring, Retraining, Promotion, and Historical Replay

The phases above (evidence-based routing, prediction logging, forecast
evaluation) are the foundation for a larger controlled MLOps lifecycle added
after this document's original sections were written. Rather than duplicate
that design here, this section is a map to where it actually lives:

```text
prediction_logs (evaluated)
  -> model_monitoring_snapshots      (rolling WAPE/bias, stable/warning/degraded)
  -> retraining_runs                 (recommendation, evidence-gated)
  -> CandidateTrainingService        (backend/src/services/candidate_training_service.py)
  -> CandidateEvaluationService      (backend/src/services/candidate_evaluation_service.py)
  -> ModelPromotionService           (backend/src/services/model_promotion_service.py)
  -> model_promotion_events          (full audit trail, promotion or rollback)
```

- **Monitoring** — [`ModelMonitoringService`](../backend/src/services/model_monitoring_service.py), full design in [docs/mlops-monitoring.md](./mlops-monitoring.md).
- **Retraining recommendation** — [`RetrainingDecisionService`](../backend/src/services/retraining_decision_service.py) — recommends only; `AUTO_RETRAIN_ENABLED=false` means nothing trains automatically.
- **Candidate training/evaluation** — evidence-gated, never auto-promoted; see [docs/model-promotion.md](./model-promotion.md).
- **Controlled promotion/rollback** — [`ModelPromotionService`](../backend/src/services/model_promotion_service.py) + [`scripts/promote_model.py`](../backend/scripts/promote_model.py) — human-run CLI only, preflight-validated (checksum, feature schema, deserialization), fully audited via `model_promotion_events`.
- **Runtime resolution** — [`runtime_model_service.py`](../backend/src/services/runtime_model_service.py) — startup prefers a valid DB-active artifact, falls back to a configured local artifact, then to a statistical method.
- **Operational cycle** — [`scripts/run_mlops_cycle.py`](../backend/scripts/run_mlops_cycle.py) — evaluates, monitors, and recommends on a schedule; never trains or promotes. See [docs/mlops-operations.md](./mlops-operations.md).
- **Historical Monitoring Replay** — [`HistoricalMonitoringReplayService`](../backend/src/services/historical_monitoring_replay_service.py) — demonstrates the monitoring pipeline against held-out historical windows because the dataset is frozen and there is no live actual-demand feed. Never writes to the live tables above; see [docs/mlops-monitoring.md](./mlops-monitoring.md) for the full distinction from live monitoring.
