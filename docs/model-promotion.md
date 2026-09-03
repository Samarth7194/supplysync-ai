# Model Promotion and Rollback

SupplySync model promotion is intentionally human controlled. Monitoring and retraining recommendation can identify that a model deserves attention, but no backend route, scheduler, or startup hook automatically promotes or rolls back an artifact.

## Lifecycle

```text
eligible candidate artifact
-> explicit operator command
-> checksum/schema/load preflight
-> candidate ACTIVE
-> previous active RETIRED
-> promotion event written
-> runtime loads DB-active artifact on restart/redeploy
```

Rollback follows the same safety rule:

```text
exact rollback artifact id
-> checksum/schema/load preflight
-> target ACTIVE
-> current active RETIRED
-> rollback event written
```

Artifacts are never deleted or overwritten by promotion or rollback.

## Commands

Run from `backend/`:

```powershell
python scripts/promote_model.py promote --artifact-id 42 --reason "approved after review"
python scripts/promote_model.py rollback --artifact-id 17 --reason "candidate underperformed"
```

Legacy flags remain accepted for compatibility:

```powershell
python scripts/promote_model.py --artifact-id 42
python scripts/promote_model.py --rollback-to-artifact-id 17
```

The CLI requires exact artifact identifiers. There is no `--latest`, `--best`, or `--auto` option.

## Promotion Evidence

A candidate cannot be promoted only because `model_artifacts.lifecycle_status` is `candidate`.

Promotion requires a persisted `retraining_runs` row where:

- `candidate_model_artifact_id` matches the requested artifact id
- `status = completed`
- `promotion_recommended = true`

The requested candidate artifact must also contain candidate evaluation metadata with:

- `promotion_eligible = true`
- candidate metrics
- active-model metrics
- compatible evaluation horizon
- enough test-point evidence

## Production Runtime Semantics

Controlled promotion: YES
DB active lifecycle switch: YES
Preflight artifact validation: YES
Rollback: YES
Startup loads exact DB-active artifact: YES
In-process atomic handoff capability: YES
Production CLI currently invokes in-process handoff: NO
Runtime switch after standalone CLI promotion: RESTART / REDEPLOY REQUIRED
Multi-worker synchronization: NO
Automatic promotion: NO

The standalone `promote_model.py` command runs outside the already-running FastAPI process. In the current Render deployment model it safely updates database lifecycle state and writes the audit event, but the running API process may continue serving its previously loaded runtime until it is restarted or redeployed. On startup, SupplySync resolves and loads the exact DB-active artifact when it is valid and available.

## Runtime Behavior

Startup resolution prefers:

```text
valid DB-active artifact
configured runtime artifact
statistical fallback
```

The in-process handoff helper is process-local. It constructs the new inventory runtime before swapping references, so future requests use the new runtime while already-running requests can finish with the old one. Multi-worker synchronization is not implemented; a restart/redeploy lets every worker load the same DB-active artifact.

## Audit Trail

`model_promotion_events` records:

- operation type: `promotion` or `rollback`
- target artifact
- previous active artifact
- retraining run reference, when promotion came from candidate evaluation
- outcome
- operator identifier
- reason
- timestamp

The audit table stores identifiers and concise reasons, not credentials or large model metadata.

## Not Implemented

- automatic promotion
- automatic rollback
- frontend promotion controls
- distributed runtime synchronization
- artifact deletion or overwrite
