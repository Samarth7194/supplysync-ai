"""Human-controlled model promotion and rollback.

Examples:
    cd backend
    python scripts/promote_model.py promote --artifact-id 7 --reason "approved candidate"
    python scripts/promote_model.py rollback --artifact-id 3 --reason "runtime issue"

Legacy forms are still accepted:
    python scripts/promote_model.py --artifact-id 7
    python scripts/promote_model.py --rollback-to-artifact-id 3

This script never trains, schedules, or automatically promotes a model. It
validates the target artifact before changing lifecycle state and never deletes
or overwrites artifact files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

from config.settings import load_settings  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from services.model_promotion_service import ModelPromotionService, ModelPromotionServiceError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote or roll back a model artifact under human control.")
    subparsers = parser.add_subparsers(dest="command")

    promote = subparsers.add_parser("promote", help="Promote an exact eligible candidate artifact.")
    promote.add_argument("--artifact-id", type=int, required=True, help="Inactive eligible candidate artifact id to promote.")
    promote.add_argument("--reason", type=str, default=None, help="Operator reason stored in model_promotion_events.")
    promote.add_argument("--initiated-by", type=str, default="manual_cli", help="Operator identifier for the audit event.")

    rollback = subparsers.add_parser("rollback", help="Roll back to an exact valid artifact id.")
    rollback.add_argument("--artifact-id", type=int, required=True, help="Artifact id to restore as active.")
    rollback.add_argument("--reason", type=str, default=None, help="Operator reason stored in model_promotion_events.")
    rollback.add_argument("--initiated-by", type=str, default="manual_cli", help="Operator identifier for the audit event.")

    legacy = parser.add_mutually_exclusive_group(required=False)
    legacy.add_argument("--artifact-id", type=int, help=argparse.SUPPRESS)
    legacy.add_argument("--rollback-to-artifact-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--reason", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--initiated-by", type=str, default="manual_cli", help=argparse.SUPPRESS)
    return parser


def _resolve_operation(args) -> tuple[str, int]:
    if args.command == "promote":
        return "promotion", args.artifact_id
    if args.command == "rollback":
        return "rollback", args.artifact_id
    if args.artifact_id is not None:
        return "promotion", args.artifact_id
    if args.rollback_to_artifact_id is not None:
        return "rollback", args.rollback_to_artifact_id
    raise ModelPromotionServiceError("Choose `promote --artifact-id ID` or `rollback --artifact-id ID`.")


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        operation, artifact_id = _resolve_operation(args)
    except ModelPromotionServiceError as exc:
        parser.error(str(exc))

    with SessionLocal() as session:
        service = ModelPromotionService(session=session, settings=load_settings())
        try:
            if operation == "promotion":
                result = service.promote_candidate(
                    artifact_id,
                    initiated_by=args.initiated_by,
                    reason=args.reason,
                )
            else:
                result = service.rollback_to_artifact(
                    artifact_id,
                    initiated_by=args.initiated_by,
                    reason=args.reason,
                )
            session.commit()
        except ModelPromotionServiceError as exc:
            session.rollback()
            print(f"Model {operation} failed safely: {exc}")
            return 1

    if not result.changed:
        print(
            f"No-op {operation}: artifact id={result.artifact.id} "
            f"model={result.artifact.model_name} version={result.artifact.version} is already active."
        )
        return 0

    print(f"Operation: {operation}")
    print(f"Target artifact: {result.artifact.id}")
    print(f"Target model: {result.artifact.model_name}")
    print(f"Target version: {result.artifact.version}")
    print(f"Previous active artifact: {result.previous_artifact.id if result.previous_artifact else None}")
    print("Preflight: checksum/schema/deserialization passed")
    print("Database lifecycle: updated and committed")
    print("Runtime handoff: not performed by this standalone CLI; restart/redeploy loads DB-active artifact")
    print(f"Event id: {result.event.id if result.event else None}")
    print(f"Event outcome: {result.event.outcome if result.event else None}")
    print("No model was trained. No artifact file was deleted or overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
