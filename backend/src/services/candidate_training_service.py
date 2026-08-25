"""Controlled candidate training orchestration for Phase F MLOps.

This service is intentionally manual. It can train and evaluate a candidate
model for an existing retraining run, but it never promotes, reloads, schedules,
or changes production inference.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sqlalchemy.orm import Session

from db.models import ModelArtifact, RetrainingRun
from features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_schema_checksum
from repositories.model_artifact_repository import ModelArtifactRepository
from repositories.retraining_repository import RetrainingRepository
from services.candidate_evaluation_service import CandidateEvaluationResult, CandidateEvaluationService
from services.model_service import ModelService


MODEL_NAME = "lightgbm_demand_forecast"


@dataclass(frozen=True)
class CandidateTrainingResult:
    retraining_run: RetrainingRun
    candidate_artifact: ModelArtifact
    evaluation: CandidateEvaluationResult
    created: bool


class CandidateTrainingError(Exception):
    """Raised when controlled candidate training cannot proceed safely."""


class CandidateTrainingService:
    def __init__(
        self,
        *,
        session: Session,
        settings: Any,
        model_dir: str | Path | None = None,
        trainer: Callable[..., Any] | None = None,
        evaluator: CandidateEvaluationService | None = None,
    ):
        self.session = session
        self.settings = settings
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[2] / "saved_models"
        self.trainer = trainer or self._default_trainer()
        self.retraining_repository = RetrainingRepository(session)
        self.artifact_repository = ModelArtifactRepository(session)
        self.evaluator = evaluator or CandidateEvaluationService(session=session, settings=settings)

    def train_candidate(
        self,
        *,
        retraining_run_id: int,
        csv_path: str | Path | None = None,
        column_mapping: dict[str, str] | None = None,
        parquet_path: str | Path | None = None,
        daily_demand: pd.DataFrame | None = None,
        horizon_days: int | None = None,
    ) -> CandidateTrainingResult:
        run = self.retraining_repository.get_retraining_run(retraining_run_id)
        if run is None:
            raise CandidateTrainingError(f"Retraining run {retraining_run_id} does not exist.")

        duplicate = self._existing_candidate(run)
        if duplicate is not None:
            evaluation = self._evaluation_from_metadata(duplicate)
            if evaluation is None:
                raise CandidateTrainingError("Existing candidate is linked but evaluation evidence is missing.")
            return CandidateTrainingResult(run, duplicate, evaluation, created=False)

        self._validate_run(run)

        active = run.baseline_model_artifact
        if active is None or not active.is_active or active.lifecycle_status != "active":
            raise CandidateTrainingError("Retraining run baseline model is not the current active model.")

        self.retraining_repository.mark_running(run)
        candidate_artifact: ModelArtifact | None = None
        try:
            artifact_file = f"{MODEL_NAME}_candidate_run_{run.id}.pkl"
            metadata_file = f"{MODEL_NAME}_candidate_run_{run.id}_metadata.json"
            training_result = self.trainer(
                csv_path=csv_path,
                column_mapping=column_mapping,
                parquet_path=parquet_path,
                model_dir=self.model_dir,
                artifact_file=artifact_file,
                metadata_file=metadata_file,
                verbose=False,
            )
            metadata = dict(training_result.metadata)
            metadata["artifact_file"] = artifact_file
            metadata["metadata_file"] = metadata_file
            metadata["model_dir"] = str(self.model_dir)
            metadata["lifecycle_status"] = "candidate"
            metadata.setdefault("candidate_training", {})
            metadata["candidate_training"].update(
                {
                    "retraining_run_id": run.id,
                    "baseline_model_artifact_id": run.baseline_model_artifact_id,
                    "source_monitoring_snapshot_id": run.source_monitoring_snapshot_id,
                    "evidence_key": run.evidence_key,
                }
            )
            self._validate_metadata(metadata)
            candidate_artifact = self.artifact_repository.register_metadata(metadata, status="candidate")
            self._validate_candidate_artifact(candidate_artifact)

            demand_frame = daily_demand if daily_demand is not None else self._load_daily_demand(parquet_path)
            evaluation = self.evaluator.evaluate(
                candidate_artifact=candidate_artifact,
                active_artifact=active,
                daily_demand=demand_frame,
                horizon_days=self._resolved_horizon(horizon_days),
                persist=True,
            )
            self.retraining_repository.mark_completed(
                run,
                candidate_model_artifact_id=candidate_artifact.id,
                promotion_recommended=evaluation.promotion_eligible,
            )
            return CandidateTrainingResult(run, candidate_artifact, evaluation, created=True)
        except Exception as exc:
            safe_reason = f"{type(exc).__name__}: {exc}"
            self.retraining_repository.mark_failed(run, reason=safe_reason)
            if candidate_artifact is not None:
                candidate_artifact.lifecycle_status = "failed"
                candidate_artifact.is_active = False
            raise CandidateTrainingError(safe_reason) from exc

    def _resolved_horizon(self, horizon_days: int | None) -> int:
        if horizon_days is not None:
            return int(horizon_days)
        inventory = getattr(self.settings, "inventory", None)
        return int(getattr(inventory, "default_lead_time_days", 7) or 7)

    def _validate_run(self, run: RetrainingRun) -> None:
        if run.status not in {"recommended", "failed"}:
            raise CandidateTrainingError(f"Retraining run status {run.status!r} cannot start candidate training.")
        if run.candidate_model_artifact_id is not None and run.status == "completed":
            raise CandidateTrainingError("Retraining run already has a completed candidate artifact.")
        if int(run.new_evaluated_forecast_days or 0) < int(self.settings.forecasting.model_retrain_min_evaluated_forecast_days):
            raise CandidateTrainingError("Retraining run does not have enough evaluated forecast-day evidence.")

    def _existing_candidate(self, run: RetrainingRun) -> ModelArtifact | None:
        if run.candidate_model_artifact_id is not None:
            return self.artifact_repository.get(run.candidate_model_artifact_id)
        if run.candidate_model_artifact is not None:
            return run.candidate_model_artifact
        duplicate_run = self.retraining_repository.candidate_run_for_evidence(run.evidence_key)
        if duplicate_run is not None and duplicate_run.candidate_model_artifact_id is not None:
            return self.artifact_repository.get(duplicate_run.candidate_model_artifact_id)
        return None

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise CandidateTrainingError("Candidate feature schema version does not match runtime schema.")
        if list(metadata.get("features") or []) != FEATURE_COLUMNS:
            raise CandidateTrainingError("Candidate feature columns do not match runtime schema.")
        if metadata.get("feature_schema_checksum") != feature_schema_checksum(FEATURE_COLUMNS):
            raise CandidateTrainingError("Candidate feature schema checksum does not match runtime schema.")
        if not metadata.get("artifact_checksum"):
            raise CandidateTrainingError("Candidate artifact checksum is missing.")

    def _validate_candidate_artifact(self, artifact: ModelArtifact) -> None:
        if artifact.lifecycle_status != "candidate" or artifact.is_active:
            raise CandidateTrainingError("Registered candidate artifact has unsafe lifecycle state.")
        if not artifact.artifact_uri or not Path(artifact.artifact_uri).exists():
            raise CandidateTrainingError("Registered candidate artifact file is missing.")
        if artifact.artifact_checksum != ModelService.checksum_file(artifact.artifact_uri):
            raise CandidateTrainingError("Registered candidate artifact checksum does not match file.")

    @staticmethod
    def _load_daily_demand(parquet_path: str | Path | None) -> pd.DataFrame:
        resolved = Path(parquet_path) if parquet_path else Path(os.environ.get("DATA_PARQUET_PATH", ""))
        if not str(resolved):
            resolved = Path(__file__).resolve().parents[3] / "data" / "processed" / "daily_demand.parquet"
        if not resolved.exists():
            raise CandidateTrainingError(f"Processed daily demand parquet is missing: {resolved}")
        return pd.read_parquet(resolved)

    @staticmethod
    def _evaluation_from_metadata(artifact: ModelArtifact) -> CandidateEvaluationResult | None:
        payload = ((artifact.training_metadata or {}).get("candidate_evaluation") or {})
        if not payload:
            return None
        return CandidateEvaluationResult(
            horizon_days=int(payload["horizon_days"]),
            test_points=int(payload["test_points"]),
            candidate_metrics=dict(payload["candidate_metrics"]),
            active_metrics=dict(payload["active_metrics"]),
            benchmark_metrics=dict(payload.get("benchmark_metrics") or {}),
            relative_wape_improvement=payload.get("relative_wape_improvement"),
            promotion_eligible=bool(payload["promotion_eligible"]),
            eligibility_reason=str(payload["eligibility_reason"]),
            temporal_split=dict(payload.get("temporal_split") or {}),
        )

    @staticmethod
    def _default_trainer() -> Callable[..., Any]:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
        spec = importlib.util.spec_from_file_location("supplysync_train_model", script_path)
        if spec is None or spec.loader is None:
            raise CandidateTrainingError("Unable to load training pipeline.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.train_lightgbm_demand_model