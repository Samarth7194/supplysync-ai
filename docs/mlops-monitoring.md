# SupplySync AI MLOps Monitoring

This document describes the current SupplySync AI MLOps monitoring layer:

- Phase A: model monitoring metrics
- Phase B: forecast performance degradation detection
- Phase C: read-only monitoring API
- Phase D: frontend model-health visibility
- Phase E: retraining recommendation and run tracking
- Historical Monitoring Replay: offline demo evidence for the same lifecycle
  (see the dedicated section below) — this is not a numbered phase and is
  never live production evidence.

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

## Historical Monitoring Replay (Demo Evidence — Not Live Monitoring)

### Why this exists

The processed dataset (`data/processed/daily_demand.parquet`) is a static
historical retail extract, 2009-12-01 through 2011-12-09. There is no
connected ERP/POS actual-demand stream. A prediction logged today targets a
future window that will never arrive in this dataset, so it can never mature
into a real `forecast_evaluations` row — live monitoring then honestly
reports `insufficient_evidence` or `unavailable` indefinitely, through no
fault of the monitoring logic itself.

Historical replay demonstrates the exact same forecast → evaluate → monitor
lifecycle honestly, by picking an anchor date inside the dataset and
evaluating against demand that has already been recorded, instead of
predictions that can never resolve.

### There are three distinct evidence classes — do not confuse them

```text
1. Offline Backtest        (scripts/evaluate_forecast.py, forecast_evaluation.json)
   One-step-ahead backtest with real actuals fed back each step. Powers the
   dashboard's "Backtest Performance" KPIs and the routing/monitoring
   baseline WAPE. Not a replay of the monitoring lifecycle itself.

2. Historical Monitoring Replay   (this section)
   A single multi-day forecast issued at a historical anchor date T using
   only demand <= T (the same hybrid routing/forecasting code production
   uses), evaluated against the already-recorded actual demand for T+1..T+H,
   then classified stable/warning/degraded with the same monitoring math as
   live snapshots. Demonstrates the operational lifecycle end-to-end.

3. Live Production Monitoring   (Phases A-C above)
   Real logged predictions evaluated against real future actuals as they
   become available. Currently limited by the lack of a live actual-demand
   feed, as described above.
```

### What replay does and does not do

It reuses production code paths: `classify_sku_demand_pattern`,
`adaptive_forecast` (the real regular/intermittent/highly-intermittent
routing — Croston-SBA and the conservative method are exercised honestly,
not hidden behind a LightGBM-only demo), and the same metric formulas
(`evaluation/metrics.py`) and classification thresholds
(`ModelMonitoringService`'s WAPE/bias rules) as live monitoring.

It never writes to `prediction_logs`, `forecast_evaluations`,
`model_monitoring_snapshots`, or `retraining_runs`. It never opens a database
session. `RetrainingDecisionService` only ever reads the live tables, so
replay evidence structurally cannot influence a live retraining
recommendation — there is no configuration flag that could accidentally
wire the two together because no shared table exists between them.

The artifact-level status (`stable`/`warning`/`degraded`) is scoped to the
`ml_lightgbm`-routed SKUs only, since Model Health specifically monitors the
active LightGBM artifact. Croston/conservative results are still reported
honestly in a separate `method_breakdown`, never folded into the
LightGBM-scoped status.

No model is trained. No model is promoted. No artifact lifecycle changes.

### Generating a replay

Run from `backend/`:

```powershell
python scripts/run_historical_monitoring_replay.py
python scripts/run_historical_monitoring_replay.py --horizon 14 --sku-limit 40 --num-windows 3
python scripts/run_historical_monitoring_replay.py --json
python scripts/run_historical_monitoring_replay.py --dry-run
```

Anchors are deterministic — non-overlapping windows walking backward from the
dataset's end date, oldest first, so results are reproducible run to run for
the same arguments and the same dataset. There is no random SKU or window
selection.

The result is written to `backend/data/historical_monitoring_replay.json` and
is read (never regenerated) by:

### `GET /api/model-monitoring/replay`

Read-only. Serves the pre-generated JSON file and always includes a
`live_monitoring` block so a caller can implement the precedence rule below
without a second request. It never triggers replay generation itself.

```json
{
  "mode": "historical_replay",
  "available": true,
  "status": "warning",
  "metrics": { "wape": 1.25, "mae": 92.2, "rmse": 228.5, "bias": 21.1, "mase": 1.23 },
  "baseline_wape": 1.0655,
  "baseline_provenance": "offline_backtest",
  "evaluation_count": 39,
  "sku_count": 59,
  "horizon_days": 7,
  "historical_period": { "start": "2011-11-19", "end": "2011-12-09" },
  "provenance": "historical_replay",
  "live_monitoring": { "available": false, "evaluation_count": 0 }
}
```

### Display precedence (frontend)

```text
1. live monitoring has a classified state (stable/warning/degraded) -> show LIVE
2. otherwise, if a historical replay is available                  -> show HISTORICAL REPLAY, clearly badged
3. otherwise                                                        -> show Unavailable
```

Live and replay evidence counts are never summed or combined. The Model
Health card renders a highly visible `HISTORICAL REPLAY` badge plus the
sentence "Based on historical holdout replay. This is not live production
monitoring." whenever replay is shown instead of live evidence.

### Honest wording

Use: historical replay, historical holdout monitoring, offline operational
replay, demo monitoring evidence.

Never describe replay as: live monitoring, production actuals, real-time
drift, or live ERP evidence.

## Implementation Status

Monitoring API: implemented.

Frontend monitoring UI: implemented.

Retraining recommendation: implemented.

Retraining decision tracking: implemented.

Candidate training and evaluation: implemented as explicit operator-controlled Phase F commands.

Controlled model promotion and rollback: implemented as explicit operator-controlled Phase G commands.

MLOps operational cycle: implemented as a scheduler-ready Phase H command.

Historical monitoring replay: implemented as an offline demo-evidence command and read-only API endpoint; never live production evidence and never wired into retraining.

Scheduled production job: not configured in repository; create it externally, for example with Render Cron.

Statistical input/data drift: not yet implemented.

Automatic retraining execution: not implemented.

Automatic promotion: not implemented.
