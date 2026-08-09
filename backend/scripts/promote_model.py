"""Promote a registered model artifact to active lifecycle status.

Usage:
    cd backend
    python scripts/promote_model.py --artifact-id 7
    python scripts/promote_model.py --artifact-id 7 --force

Without ``--force``, the artifact must have at least one persisted
forecast_evaluations row. ``--force`` is an explicit local/admin override and
prints that evidence checks were bypassed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

from db.session import SessionLocal  # noqa: E402
from repositories.model_artifact_repository import (  # noqa: E402
    ModelArtifactRepository,
    ModelPromotionError,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a registered model artifact.")
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass evaluation-evidence requirement. Intended for explicit local/admin override only.",
    )
    args = parser.parse_args()

    with SessionLocal() as session:
        repo = ModelArtifactRepository(session)
        try:
            artifact = repo.promote(args.artifact_id, force=args.force)
            session.commit()
        except ModelPromotionError:
            session.rollback()
            raise

    override = " (forced override; evidence check bypassed)" if args.force else ""
    print(
        "Promoted artifact "
        f"id={artifact.id} model={artifact.model_name} version={artifact.version} "
        f"status={artifact.lifecycle_status}{override}"
    )


if __name__ == "__main__":
    main()
