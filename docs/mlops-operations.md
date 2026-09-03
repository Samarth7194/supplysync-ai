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

## Known Limitations

- Recursive LightGBM future calendar-feature advancement limitation remains.
- Live ERP/POS ingestion is not implemented.
- In-process runtime handoff is process-local and tested, but the standalone CLI does not mutate an already-running Render/FastAPI process.
- Automatic retraining is disabled.
- Automatic promotion is unavailable.
- Runtime switch after standalone CLI promotion requires restart/redeploy.
- Scheduled production cycle must be configured externally.
- Artifact availability across redeploys depends on the production artifact strategy.
