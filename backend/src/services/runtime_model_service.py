"""Runtime model artifact resolution and safe loading.

Phase G keeps runtime loading explicit: prefer a valid DB-active artifact,
fall back to the configured runtime artifact, and finally allow statistical
forecasting when no ML artifact can be loaded.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from db.models import ModelArtifact
from repositories.model_artifact_repository import ModelArtifactRepository
from services.model_service import ModelArtifactValidationError, ModelService
from features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_schema_checksum


MODEL_NAME = "lightgbm_demand_forecast"
BACKEND_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoadedRuntimeModel:
    model: Any | None
    artifact: ModelArtifact | None
    model_name: str
    model_version: str | None
    feature_columns: list[str] | None
    status: dict[str, Any]


class RuntimeModelLoadError(Exception):
    """Raised when an artifact cannot be loaded safely for inference."""


def _portable_name(path_value: str | None) -> str | None:
    if not path_value:
        return None
    normalized = str(path_value).replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def resolve_artifact_path(artifact_uri: str | None, *, model_dir: str | Path) -> Path | None:
    """Resolve persisted artifact URIs against the current deployment layout.

    Older rows may contain developer-machine absolute paths. In production the
    artifact zip is restored under backend/saved_models, so the safe portable
    fallback is the original filename under the configured model directory.
    """
    if not artifact_uri:
        return None

    raw = Path(str(artifact_uri))
    candidates: list[Path] = []
    candidates.append(raw)
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, BACKEND_DIR / raw, Path(model_dir) / raw])

    filename = _portable_name(artifact_uri)
    if filename:
        candidates.extend([Path(model_dir) / filename, BACKEND_DIR / "saved_models" / filename])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def validate_artifact_record(artifact: ModelArtifact, *, model_dir: str | Path) -> Path:
    if artifact.model_type != "ml":
        raise RuntimeModelLoadError("Only ML artifacts can be loaded for runtime inference.")
    path = resolve_artifact_path(artifact.artifact_uri, model_dir=model_dir)
    if path is None or not path.exists():
        raise RuntimeModelLoadError("Artifact file is missing or not portable to this runtime.")
    if artifact.artifact_checksum:
        checksum = ModelService.checksum_file(path)
        if checksum != artifact.artifact_checksum:
            raise RuntimeModelLoadError("Artifact checksum does not match registered checksum.")
    if artifact.feature_schema_version and artifact.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise RuntimeModelLoadError("Artifact feature schema version does not match runtime schema.")
    if artifact.feature_schema and list(artifact.feature_schema) != FEATURE_COLUMNS:
        raise RuntimeModelLoadError("Artifact feature columns do not match runtime schema.")
    if artifact.feature_schema_checksum and artifact.feature_schema_checksum != feature_schema_checksum(FEATURE_COLUMNS):
        raise RuntimeModelLoadError("Artifact feature schema checksum does not match runtime schema.")
    return path


def load_model_artifact(artifact: ModelArtifact, *, model_dir: str | Path) -> LoadedRuntimeModel:
    path = validate_artifact_record(artifact, model_dir=model_dir)
    try:
        with path.open("rb") as fh:
            model = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001 - surface as controlled runtime load failure
        raise RuntimeModelLoadError(f"Artifact could not be loaded: {type(exc).__name__}") from exc

    features = list(artifact.feature_schema or FEATURE_COLUMNS)
    status = {
        "valid": True,
        "source": "db_active_artifact",
        "artifact_id": artifact.id,
        "model_name": artifact.model_name,
        "version": artifact.version,
        "feature_schema_version": artifact.feature_schema_version,
        "lifecycle_status": artifact.lifecycle_status,
        "artifact_uri": str(path),
        "loadable": True,
    }
    return LoadedRuntimeModel(
        model=model,
        artifact=artifact,
        model_name=artifact.model_name,
        model_version=artifact.version,
        feature_columns=features,
        status=status,
    )


def load_runtime_model(
    *,
    settings: Any,
    session_factory: Callable[[], Session] | None = None,
    model_name: str = MODEL_NAME,
) -> LoadedRuntimeModel:
    model_dir = settings.forecasting.model_path
    db_error: str | None = None
    if session_factory is not None:
        try:
            with session_factory() as session:
                artifact = ModelArtifactRepository(session).active_for_name(model_name)
                if artifact is not None:
                    try:
                        return load_model_artifact(artifact, model_dir=model_dir)
                    except RuntimeModelLoadError as exc:
                        db_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - startup must retain safe fallback
            db_error = f"{type(exc).__name__}: {exc}"

    service = ModelService(model_dir=str(model_dir))
    try:
        status = service.artifact_status(model_name)
        model = service.load_model(model_name)
        metadata = service.get_model_metadata(model_name) or {}
        status.update({"source": "configured_runtime_artifact", "artifact_id": None, "loadable": True})
        if db_error:
            status["db_active_error"] = db_error
        return LoadedRuntimeModel(
            model=model,
            artifact=None,
            model_name=model_name,
            model_version=metadata.get("version"),
            feature_columns=metadata.get("features"),
            status=status,
        )
    except (FileNotFoundError, ModelArtifactValidationError, RuntimeModelLoadError) as exc:
        return LoadedRuntimeModel(
            model=None,
            artifact=None,
            model_name=model_name,
            model_version=None,
            feature_columns=None,
            status={
                "valid": False,
                "source": "statistical_fallback",
                "artifact_id": None,
                "model_name": model_name,
                "version": None,
                "feature_schema_version": None,
                "lifecycle_status": None,
                "loadable": False,
                "error": str(exc),
                **({"db_active_error": db_error} if db_error else {}),
            },
        )
