# SupplySync AI MLOps Monitoring

This document describes the current SupplySync AI MLOps monitoring layer:

- Phase A: model monitoring metrics
- Phase B: forecast performance degradation detection
- Phase C: read-only monitoring API
- Phase D: frontend model-health visibility
- Phase E: retraining recommendation and run tracking

## What Phase A Does

Phase A creates a backend-only monitoring foundation. It summarizes completed
forecast evaluations into `model_monitoring_snapshots`.

The monitoring snapshot answers:

- how many completed evaluations exist for the current model scope
- recent WAPE, MAE, RMSE, bias, and MASE
- residual mean and residual standard deviation when daily actuals can be
  reconstructed
- baseline WAPE and where that baseline came from
- whether there is enough evidence to treat the snapshot as usable data
- whether recent forecast performance is stable, warning, or degraded

## Evidence Source

Monitoring uses only completed `forecast_evaluations` rows linked to
`prediction_logs`.

It does not compute metrics from unevaluated predictions. A prediction becomes
eligible only after its forecast horizon has completed and the evaluation
service has matched it to actual demand.

## Rolling Window

Default window:

```text
latest 30 completed evaluations
```

Default lookback cap:

```text
90 days
```

Default minimum evidence:

```text
30 completed evaluations
```

These values are configured through:

```text
MODEL_MONITORING_ENABLED=true
MODEL_MONITORING_WINDOW_EVALUATIONS=30
MODEL_MONITORING_LOOKBACK_DAYS=90
MODEL_MONITORING_MIN_EVALUATIONS=30
```

## Metrics

The snapshot stores:

- WAPE
- MAE
- RMSE
- bias
- MASE
- evaluation count
- residual mean
- residual standard deviation

WAPE is not averaged blindly when actual observed demand is available. The
service aggregates absolute error over total actual demand. MAE and bias are
weighted by test-point count. RMSE is aggregated from squared RMSE weighted by
test-point count. MASE is weighted by test-point count when valid MASE values
exist.

Residuals use:

```text
actual - predicted
```

Residual standard deviation uses sample standard deviation and is `null` when
fewer than two residual observations are available.

## Model Scope

Monitoring prefers exact `model_artifact_id` scoping.

Fallback order:

```text
model_artifact_id
model_name + model_version
model_name
```

The goal is to avoid mixing evidence from unrelated model versions.

## Baseline Provenance

Baseline WAPE is labeled by provenance:

```text
promotion_evidence
artifact_metadata
offline_backtest
unavailable
```

Offline backtest evidence is never labeled as production performance.

## Insufficient Evidence

If fewer than `MODEL_MONITORING_MIN_EVALUATIONS` completed evaluations exist,
the script still creates or returns a snapshot with:

```text
status = insufficient_evidence
```

This is not an operational failure.

## Phase B Performance Degradation Detection

Phase B compares recent aggregate WAPE against the selected baseline WAPE.

Default thresholds:

```text
MODEL_MONITORING_WAPE_WARNING_THRESHOLD=0.15
MODEL_MONITORING_WAPE_DEGRADATION_THRESHOLD=0.25
MODEL_MONITORING_BIAS_WARNING_RATIO=0.20
MODEL_MONITORING_DEGRADATION_CONSECUTIVE_RUNS=2
```

The WAPE thresholds are relative changes:

```text
recent_wape >= baseline_wape * 1.15 -> warning
recent_wape >= baseline_wape * 1.25 -> degradation candidate
```

A single degradation-threshold breach is not enough to mark the model
`degraded`. The breach must persist across consecutive monitoring snapshots
with newer evidence for the same model scope.

States:

```text
insufficient_evidence
stable
warning
degraded
```

Deterministic reasons include:

```text
wape_within_baseline
wape_warning_threshold_exceeded
wape_degradation_threshold_exceeded
persistent_wape_degradation
bias_warning
baseline_unavailable
baseline_zero
insufficient_evidence
```

Bias is a supporting warning signal only. It is normalized as:

```text
bias_ratio = bias / mean_actual_demand
```

Bias alone does not mark a model degraded.

This is forecast performance degradation detection. It is not feature drift,
covariate drift, concept drift, or statistical input/data drift.

## Script

Run from `backend/`:

```powershell
python scripts/run_model_monitoring.py
```

The script opens a normal SQLAlchemy session, creates a monitoring snapshot,
prints a concise summary, and exits successfully when evidence is insufficient.

## Monitoring API

Phase C exposes backend monitoring results through typed FastAPI endpoints.

### `GET /api/model-monitoring`

Returns the latest monitoring snapshot for the current model scope.

If no snapshot exists, the endpoint returns `200 OK` with:

```json
{
  "model_name": "lightgbm_demand_forecast",
  "status": "unavailable",
  "degradation_reason": "monitoring_unavailable",
  "degradation_message": "No model monitoring snapshot has been created yet."
}
```

Example monitored response:

```json
{
  "model_artifact_id": 7,
  "model_name": "lightgbm_demand_forecast",
  "model_version": "lightgbm_demand_forecast-...",
  "lifecycle_status": "active",
  "status": "warning",
  "degradation_reason": "wape_degradation_threshold_exceeded",
  "evaluation_count": 30,
  "metric_wape": 1.3,
  "baseline_wape": 1.0,
  "baseline_provenance": "offline_backtest",
  "wape_relative_change": 0.3,
  "bias_ratio": 0.01
}
```

### `GET /api/model-monitoring/history`

Returns recent monitoring snapshots newest first.

Supported query parameters:

```text
limit              default 20, max 100
model_artifact_id  optional exact artifact filter
status             optional state filter
```

### `POST /api/model-monitoring/evaluate`

Runs one monitoring evaluation using completed forecast evaluations and returns
the created or idempotently reused snapshot.

This endpoint:

- does NOT retrain
- does NOT promote
- does NOT switch model
- does NOT change forecasting behavior
- does NOT change inventory formulas

## Frontend Model Health

Phase D surfaces the latest monitoring snapshot on the main dashboard as a
compact read-only `Model Health` card. The card shows status, current model,
lifecycle status, recent WAPE, baseline WAPE, relative WAPE change, bias ratio,
evaluation count, baseline provenance, last monitoring time, and the backend
reason/message when available.

The frontend labels this as forecast performance monitoring, not data drift.
If the baseline provenance is `offline_backtest`, the UI explicitly states
that the baseline comes from offline evaluation rather than live production
history.

The monitoring request is non-blocking. If `/api/model-monitoring` is
temporarily unavailable, the dashboard still loads SKU data, KPIs, stock
management, and recommendations.

## Phase E Retraining Recommendation

Phase E answers one question:

```text
Should retraining be recommended for human review?
```

It does not train a model, create a candidate artifact, evaluate a candidate,
promote a model, switch inference, or schedule background work.

Retraining recommendation uses:

- the active model artifact
- the latest monitoring snapshot for that artifact
- completed logged forecast evaluations for that same artifact
- previous retraining recommendation or attempt records
- configured minimum new evidence and cooldown windows

Default configuration:

```text
AUTO_RETRAIN_ENABLED=false
MODEL_RETRAIN_MIN_EVALUATED_FORECAST_DAYS=100
MODEL_RETRAIN_COOLDOWN_DAYS=14
MODEL_RETRAIN_REQUIRE_DEGRADED_STATUS=true
```

`AUTO_RETRAIN_ENABLED=false` does not suppress recommendation calculation. It
means the system may recommend retraining but will not automatically execute a
training job.

Default recommendation conditions:

```text
latest monitoring status == degraded
AND current active model artifact is identified
AND new completed evaluated forecast-days >= minimum required
AND cooldown since last retraining recommendation/attempt is satisfied
```

The service does not recommend retraining for:

```text
unavailable
insufficient_evidence
stable
warning
```

Bias warning alone does not trigger retraining.

### New Evaluated Forecast-Days

Phase E counts completed forecast-day evidence from logged evaluations scoped to
the current model artifact. It uses:

```text
n_test_points
then horizon_days
then prediction_log.forecast_horizon_days
```

This avoids treating a 1-row 30-day forecast the same as a 1-row 1-day
forecast. Evidence from previous model artifacts does not count toward the
current model.

### Retraining Runs

Recommendations are tracked in `retraining_runs`.

Important fields:

```text
triggered_at
trigger_reason
status
baseline_model_artifact_id
source_monitoring_snapshot_id
new_evaluated_forecast_days
candidate_model_artifact_id
promotion_recommended
evidence_key
```

In Phase E, a persisted recommendation normally has:

```text
status = recommended
promotion_recommended = false
candidate_model_artifact_id = null
```

The `evidence_key` prevents duplicate recommendation records for the same model
and monitoring evidence.

Deterministic reasons include:

```text
model_unavailable
monitoring_unavailable
monitoring_not_degraded
insufficient_new_evidence
cooldown_active
retraining_recommended
```

### Retraining Status API

`GET /api/model-retraining/status` returns read-only recommendation visibility:

```json
{
  "recommended": true,
  "reason": "retraining_recommended",
  "latest_monitoring_status": "degraded",
  "new_evaluated_forecast_days": 126,
  "minimum_required": 100,
  "cooldown_days": 14,
  "cooldown_remaining_days": 0
}
```

The API does not persist a recommendation record. The script below is the
explicit operational command for persistence.

### Script

Run from `backend/`:

```powershell
python scripts/check_retraining_recommendation.py
```

The script evaluates the latest monitoring state and persists or reuses a
recommendation record when the evidence qualifies. It never calls
`train_model.py`.

A retraining recommendation does not change the model serving production
inference.

## Implementation Status

Monitoring API: implemented.

Frontend monitoring UI: implemented.

Retraining recommendation: implemented.

Retraining decision tracking: implemented.

Statistical input/data drift: not yet implemented.

Automatic retraining: not yet implemented.

Candidate evaluation: not yet implemented.

Automatic promotion: not implemented.

Scheduled execution: not yet implemented.
