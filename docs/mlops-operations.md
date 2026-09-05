# MLOps Operations

SupplySync supports a production-safe MLOps operating loop that can be run manually now and scheduled later.

## Safe Operational Cycle

Run from `backend/`:

```powershell
python scripts/run_mlops_cycle.py
```

Optional dry run:

```powershell
python scripts/run_mlops_cycle.py --dry-run
```

JSON output for automation:

```powershell
python scripts/run_mlops_cycle.py --json
```

The cycle performs:

```text
evaluate due prediction logs
-> create or reuse model monitoring snapshot
-> classify performance as insufficient_evidence, stable, warning, or degraded
-> create or reuse retraining recommendation when evidence qualifies
-> stop
```

It does not train a candidate model. It does not promote a candidate model. It does not roll back a model. It does not deploy artifacts.

## Scheduling Readiness

Recommended cadence: daily.

Recommended Render Cron command:

```bash
cd backend && python scripts/run_mlops_cycle.py
```

Required environment variables are the same backend runtime variables used by the API service, including `DATABASE_URL` and artifact/data configuration. Do not put secret values in the repository.

Use Render Cron or another production scheduler rather than GitHub Actions for production monitoring because the job needs production database access. CI should validate code; it should not become the production monitoring control plane.

## Operator Status

Run from `backend/`:

```powershell
python scripts/mlops_status.py
```

This command is read-only. It prints:

- runtime model source and version
- DB active model
- latest monitoring state
- latest WAPE and baseline WAPE
- retraining recommendation status
- latest candidate artifact, if any
- latest promotion or rollback event

## Current Lifecycle

```text
prediction
-> logged prediction evaluation
-> model monitoring
-> performance degradation classification
-> retraining recommendation
-> controlled candidate training
-> candidate evaluation
-> explicit promotion
-> restart/redeploy loads DB-active artifact into the running API
-> explicit rollback if needed
```

Automatic steps when the cycle is scheduled:

- prediction evaluation
- monitoring snapshot creation
- degradation classification
- retraining recommendation

Manual/controlled steps:

- candidate training
- candidate promotion
- rollback

`AUTO_RETRAIN_ENABLED=false` remains the intended production posture. The system may recommend retraining, but it does not execute retraining automatically.

## Historical Monitoring Replay (Not Live Monitoring)

The processed dataset is historical (2009-12-01 through 2011-12-09) with no
connected ERP/POS actual-demand stream, so predictions logged today target
windows that will never resolve into real `forecast_evaluations`. Live
monitoring honestly reports `insufficient_evidence`/`unavailable` in that
situation — it is not broken, it is correctly refusing to fabricate evidence.

`scripts/run_historical_monitoring_replay.py` demonstrates the same
evaluate → monitor lifecycle against held-out historical windows instead,
reusing the real hybrid forecasting/routing and monitoring-classification
code. It never writes to any live table (`prediction_logs`,
`forecast_evaluations`, `model_monitoring_snapshots`, `retraining_runs`),
never opens a database session, and its output is always tagged
`provenance: historical_replay`. `GET /api/model-monitoring/replay` serves
the pre-generated result read-only and is never used as a substitute for
live evidence in `RetrainingDecisionService`. See
`docs/mlops-monitoring.md` for the full design and the three-way distinction
between offline backtest, historical replay, and live monitoring.

Run from `backend/`:

```powershell
python scripts/run_historical_monitoring_replay.py
```

## Known Limitations

- Recursive LightGBM future calendar-feature advancement limitation remains.
- Live ERP/POS ingestion is not implemented.
- In-process runtime handoff is process-local and tested, but the standalone CLI does not mutate an already-running Render/FastAPI process.
- Automatic retraining is disabled.
- Automatic promotion is unavailable.
- Runtime switch after standalone CLI promotion requires restart/redeploy.
- Scheduled production cycle must be configured externally.
- Artifact availability across redeploys depends on the production artifact strategy.
