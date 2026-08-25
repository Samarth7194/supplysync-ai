"""Repository for model artifact registration and lifecycle changes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import ForecastEvaluation, ModelArtifact
from services.model_service import ModelService


class ModelPromotionError(Exception):
    """Raised when an artifact cannot be promoted safely."""


class ModelArtifactRepository:
    """Owns model artifact identity, registration, promotion, and rollback."""

    def __init__(self, session: Session):
        self.session = session

    def get(self, artifact_id: int) -> ModelArtifact | None:
        return self.session.get(ModelArtifact, artifact_id)

    def active_for_name(self, model_name: str) -> ModelArtifact | None:
        stmt = (
            select(ModelArtifact)
            .where(
                ModelArtifact.model_name == model_name,
                ModelArtifact.lifecycle_status == "active",
                ModelArtifact.is_active.is_(True),
            )
            .order_by(ModelArtifact.activated_at.desc().nullslast(), ModelArtifact.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def register_metadata(self, metadata: Mapping[str, Any], *, status: str = "candidate") -> ModelArtifact:
        """Create or update a model_artifacts row from artifact-side metadata."""
        if status not in {"candidate", "active", "retired", "failed"}:
            raise ValueError(f"Unsupported lifecycle status: {status}")

        model_name = str(metadata["model_name"])
        version = str(metadata["version"])
        stmt = select(ModelArtifact).where(
            ModelArtifact.model_name == model_name,
            ModelArtifact.version == version,
        )
        artifact = self.session.scalar(stmt)
        payload = self._values_from_metadata(metadata, status=status)

        if artifact is None:
            artifact = ModelArtifact(**payload)
            self.session.add(artifact)
        else:
            for key, value in payload.items():
                setattr(artifact, key, value)

        self.session.flush()
        return artifact

    def promote(self, artifact_id: int, *, force: bool = False) -> ModelArtifact:
        """Activate an artifact and retire previous active artifacts atomically."""
        artifact = self.get(artifact_id)
        if artifact is None:
            raise ModelPromotionError(f"Model artifact {artifact_id} does not exist.")
        if artifact.model_type != "ml":
            raise ModelPromotionError("Only ML artifacts can be promoted through the model registry.")
        if artifact.lifecycle_status == "failed":
            raise ModelPromotionError("Failed artifacts cannot be promoted.")
        if not artifact.artifact_uri or not Path(artifact.artifact_uri).exists():
            raise ModelPromotionError("Artifact file is missing.")
        if artifact.artifact_checksum:
            checksum = ModelService.checksum_file(artifact.artifact_uri)
            if checksum != artifact.artifact_checksum:
                raise ModelPromotionError("Artifact checksum does not match the registered checksum.")
        if not force and not self._has_evaluation(artifact.id):
            raise ModelPromotionError("Promotion requires evaluation evidence. Use --force for an explicit override.")

        now = datetime.now(timezone.utc)
        active_stmt = select(ModelArtifact).where(
            ModelArtifact.model_name == artifact.model_name,
            ModelArtifact.is_active.is_(True),
        )
        for current in self.session.scalars(active_stmt):
            if current.id == artifact.id:
                continue
            current.is_active = False
            current.lifecycle_status = "retired"
            current.retired_at = now

        artifact.is_active = True
        artifact.lifecycle_status = "active"
        artifact.activated_at = now
        artifact.retired_at = None
        self.session.flush()
        return artifact

    def retire(self, artifact_id: int) -> ModelArtifact:
        artifact = self.get(artifact_id)
        if artifact is None:
            raise ModelPromotionError(f"Model artifact {artifact_id} does not exist.")
        artifact.is_active = False
        artifact.lifecycle_status = "retired"
        artifact.retired_at = datetime.now(timezone.utc)
        self.session.flush()
        return artifact

    def _has_evaluation(self, artifact_id: int) -> bool:
        stmt = select(func.count(ForecastEvaluation.id)).where(ForecastEvaluation.model_artifact_id == artifact_id)
        return int(self.session.scalar(stmt) or 0) > 0

    @staticmethod
    def _values_from_metadata(metadata: Mapping[str, Any], *, status: str) -> dict[str, Any]:
        model_name = str(metadata["model_name"])
        artifact_file = metadata.get("artifact_file") or f"{model_name}.pkl"
        model_dir = Path(str(metadata.get("model_dir") or "backend/saved_models"))
        artifact_path = Path(str(artifact_file))
        if not artifact_path.is_absolute():
            artifact_path = model_dir / artifact_path
        metadata_file = metadata.get("metadata_file") or f"{model_name}_metadata.json"
        metadata_path = model_dir / str(metadata_file)

        return {
            "model_name": model_name,
            "model_family": metadata.get("model_family"),
            "model_type": metadata.get("model_type", "ml"),
            "version": metadata["version"],
            "artifact_checksum": metadata.get("artifact_checksum"),
            "checksum_algorithm": metadata.get("checksum_algorithm", "sha256"),
            "artifact_uri": str(artifact_path),
            "metadata_uri": str(metadata_path),
            "feature_schema": metadata.get("features"),
            "feature_schema_version": metadata.get("feature_schema_version"),
            "feature_schema_checksum": metadata.get("feature_schema_checksum"),
            "training_dataset": metadata.get("dataset"),
            "training_finished_at": _parse_datetime(metadata.get("saved_at")),
            "training_metrics": {
                "mae": metadata.get("mae"),
                "rmse": metadata.get("rmse"),
                "mape": metadata.get("mape"),
            },
            "training_metadata": {
                "train_skus": metadata.get("train_skus"),
                "n_train_rows": metadata.get("n_train_rows"),
                "n_test_rows": metadata.get("n_test_rows"),
                "training_data": metadata.get("training_data"),
                "training_config": metadata.get("training_config"),
                "candidate_training": metadata.get("candidate_training"),
                "candidate_evaluation": metadata.get("candidate_evaluation"),
            },
            "lifecycle_status": status,
            "is_active": status == "active",
        }


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
