"""Register an existing saved model metadata file as a candidate artifact.

Usage:
    cd backend
    python scripts/register_model_artifact.py --model-name lightgbm_demand_forecast
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

from db.session import SessionLocal  # noqa: E402
from repositories.model_artifact_repository import ModelArtifactRepository  # noqa: E402
from services.model_service import ModelService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a saved model artifact.")
    parser.add_argument("--model-name", default="lightgbm_demand_forecast")
    parser.add_argument("--model-dir", default=str(BACKEND_DIR / "saved_models"))
    args = parser.parse_args()

    service = ModelService(model_dir=args.model_dir)
    metadata = service.validate_model_artifact(args.model_name)
    metadata["model_dir"] = args.model_dir

    with SessionLocal() as session:
        artifact = ModelArtifactRepository(session).register_metadata(metadata, status="candidate")
        session.commit()

    print(
        "Registered candidate artifact "
        f"id={artifact.id} model={artifact.model_name} version={artifact.version} "
        f"checksum={artifact.artifact_checksum}"
    )


if __name__ == "__main__":
    main()
