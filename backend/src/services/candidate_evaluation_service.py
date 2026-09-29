"""Controlled candidate evaluation for Phase F MLOps.

Compares a candidate LightGBM artifact against the current active LightGBM
artifact and the shared reference baselines using the same rolling-origin,
multi-step backtest as every other evaluation in the project
(``evaluation.backtest``). It never promotes, reloads, or changes production
inference.

Scope note: LightGBM only serves *regular*-demand SKUs in production, so the
comparison is scored on the ``regular`` demand class, not on all SKUs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sqlalchemy.orm import Session

from db.models import ForecastEvaluation, ModelArtifact
from evaluation import backtest as bt


MODEL_NAME = "lightgbm_demand_forecast"
CANDIDATE_METHOD = "candidate_lightgbm"
ACTIVE_METHOD = "active_lightgbm"
BIAS_WORSENING_TOLERANCE = 0.20
EVALUATED_DEMAND_CLASS = "regular"
# The decision consumes the horizon *total* (lead-time demand), so that is the
# metric the promotion gate compares.
GATE_METRIC = "wape_lead_time_sum"


@dataclass(frozen=True)
class CandidateEvaluationResult:
    horizon_days: int
    test_points: int
    candidate_metrics: dict[str, Any]
    active_metrics: dict[str, Any]
    benchmark_metrics: dict[str, dict[str, Any]]
    relative_wape_improvement: float | None
    promotion_eligible: bool
    eligibility_reason: str
    temporal_split: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_days": self.horizon_days,
            "test_points": self.test_points,
            "candidate_metrics": self.candidate_metrics,
            "active_metrics": self.active_metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "relative_wape_improvement": self.relative_wape_improvement,
            "promotion_eligible": self.promotion_eligible,
            "eligibility_reason": self.eligibility_reason,
            "temporal_split": self.temporal_split,
        }


class CandidateEvaluationError(Exception):
    """Raised when candidate evaluation cannot be completed safely."""


class CandidateEvaluationService:
    def __init__(
        self,
        *,
        session: Session | None = None,
        settings: Any | None = None,
        model_loader: Callable[[ModelArtifact], Any] | None = None,
    ):
        self.session = session
        self.settings = settings
        self.model_loader = model_loader or self._load_model

    def evaluate(
        self,
        *,
        candidate_artifact: ModelArtifact,
        active_artifact: ModelArtifact,
        daily_demand: pd.DataFrame,
        horizon_days: int = 30,
        top_n: int = 100,
        eval_days: int | None = None,
        persist: bool = True,
    ) -> CandidateEvaluationResult:
        if candidate_artifact.lifecycle_status != "candidate" or candidate_artifact.is_active:
            raise CandidateEvaluationError("Candidate artifact must be non-active with candidate lifecycle status.")
        if not active_artifact.is_active or active_artifact.lifecycle_status != "active":
            raise CandidateEvaluationError("Active comparison artifact is unavailable.")
        if horizon_days < 1:
            raise CandidateEvaluationError("Evaluation horizon must be positive.")

        candidate_columns = list(candidate_artifact.feature_schema or [])
        if not candidate_columns:
            raise CandidateEvaluationError("Candidate artifact has no feature schema.")
        active_columns = list(active_artifact.feature_schema or candidate_columns)

        eval_days = int(eval_days or max(30, horizon_days))
        leakage_reason = self._training_overlap(candidate_artifact, eval_days)

        candidate_model = self.model_loader(candidate_artifact)
        active_model = self.model_loader(active_artifact)
        backtest = self._run_backtest(
            daily_demand=daily_demand,
            forecasters={
                CANDIDATE_METHOD: bt.lightgbm_forecaster(candidate_model, candidate_columns),
                ACTIVE_METHOD: bt.lightgbm_forecaster(active_model, active_columns),
            },
            horizon_days=horizon_days,
            top_n=top_n,
            eval_days=eval_days,
        )

        scoped = backtest.aggregates.get(EVALUATED_DEMAND_CLASS)
        if not scoped:
            raise CandidateEvaluationError(
                f"No {EVALUATED_DEMAND_CLASS}-demand origins were available; LightGBM cannot be evaluated."
            )
        candidate_metrics = scoped.get(CANDIDATE_METHOD)
        active_metrics = scoped.get(ACTIVE_METHOD)
        if candidate_metrics is None or active_metrics is None:
            raise CandidateEvaluationError("Candidate and active LightGBM metrics are required.")

        benchmark_metrics = {k: v for k, v in scoped.items() if k not in {CANDIDATE_METHOD, ACTIVE_METHOD}}
        relative = self._relative_wape_improvement(candidate_metrics, active_metrics)
        eligible, reason = self._promotion_eligibility(
            candidate_metrics, active_metrics, relative, horizon_days, benchmark_metrics
        )
        if leakage_reason and eligible:
            eligible, reason = False, leakage_reason

        result = CandidateEvaluationResult(
            horizon_days=horizon_days,
            test_points=int(candidate_metrics.get("n_test_points") or 0),
            candidate_metrics=candidate_metrics,
            active_metrics=active_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_wape_improvement=relative,
            promotion_eligible=eligible,
            eligibility_reason=reason,
            temporal_split=self._temporal_split(backtest, eval_days),
        )

        candidate_artifact.training_metadata = dict(candidate_artifact.training_metadata or {})
        candidate_artifact.training_metadata["candidate_evaluation"] = result.as_dict()
        candidate_artifact.training_metrics = dict(candidate_artifact.training_metrics or {})
        candidate_artifact.training_metrics.update({
            "candidate_wape": candidate_metrics.get("wape"),
            "candidate_wape_lead_time_sum": candidate_metrics.get("wape_lead_time_sum"),
            "candidate_mae": candidate_metrics.get("mae"),
            "candidate_rmse": candidate_metrics.get("rmse"),
            "candidate_bias": candidate_metrics.get("bias"),
            "candidate_mase": candidate_metrics.get("mase"),
            "active_wape": active_metrics.get("wape"),
            "relative_wape_improvement": relative,
            "promotion_eligible": eligible,
        })

        if persist:
            if self.session is None:
                raise CandidateEvaluationError("A SQLAlchemy session is required to persist candidate evaluation evidence.")
            self._persist_aggregate_evaluations(candidate_artifact, active_artifact, result, scoped)

        return result

    @staticmethod
    def _training_overlap(candidate: ModelArtifact, eval_days: int) -> str | None:
        """Block promotion when the evaluation window was not held out from training."""
        config = ((candidate.training_metadata or {}).get("training_config") or {})
        holdout = config.get("holdout_days")
        if holdout is None:
            return None
        if eval_days > int(holdout):
            return f"evaluation_window_overlaps_training:{eval_days}>{int(holdout)}"
        return None

    @staticmethod
    def _run_backtest(
        *,
        daily_demand: pd.DataFrame,
        forecasters: dict[str, Any],
        horizon_days: int,
        top_n: int,
        eval_days: int,
    ) -> bt.BacktestResult:
        required = {"StockCode", "date", "demand"}
        missing = required - set(daily_demand.columns)
        if missing:
            raise CandidateEvaluationError(f"Daily demand data missing required columns: {sorted(missing)}")

        daily = daily_demand.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        dataset_end = daily["date"].max()
        cutoff = dataset_end - pd.Timedelta(days=eval_days)
        skus = bt.select_eval_skus(daily, cutoff=cutoff, min_active_days=60, max_skus=top_n)
        if not skus:
            raise CandidateEvaluationError("No SKUs produced an evaluable temporal holdout.")
        series = bt.prepare_sku_series(daily, skus=skus, pad_to=dataset_end)
        try:
            config = bt.BacktestConfig(horizon=horizon_days, eval_days=eval_days)
        except ValueError as exc:
            raise CandidateEvaluationError(str(exc)) from exc
        result = bt.run_backtest(series, forecasters, config, dataset_end=dataset_end)
        if result.n_origins == 0:
            raise CandidateEvaluationError("No SKUs produced an evaluable temporal holdout.")
        return result

    def _promotion_eligibility(
        self,
        candidate: dict[str, Any],
        active: dict[str, Any],
        relative_improvement: float | None,
        horizon_days: int,
        benchmark_metrics: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        min_points = int(getattr(getattr(self.settings, "forecasting", None), "routing_min_evaluation_points", 100) or 100)
        min_improvement = float(getattr(getattr(self.settings, "forecasting", None), "routing_min_relative_improvement", 0.05) or 0.05)
        test_points = int(candidate.get("n_test_points") or 0)
        if test_points < min_points:
            return False, f"insufficient_test_points:{test_points}<{min_points}"
        if not self._horizon_compatible(horizon_days):
            expected = self._expected_horizon()
            return False, f"incompatible_horizon:{horizon_days}!={expected}" if expected else "invalid_horizon"
        if relative_improvement is None:
            return False, "wape_comparison_unavailable"
        if relative_improvement < min_improvement:
            return False, f"wape_improvement_below_threshold:{relative_improvement:.4f}<{min_improvement:.4f}"
        if self._bias_materially_worse(candidate.get("bias"), active.get("bias")):
            return False, "candidate_bias_materially_worse"
        # Beating the active artifact is necessary but not sufficient: an active
        # model can itself be bad (this project's own history is the proof —
        # see docs/model-candidates-comparison.md). The candidate must also beat
        # the strongest reference baseline for this class, not just whatever is
        # currently deployed.
        baseline = bt.strongest_baseline(benchmark_metrics or {}, metric=GATE_METRIC)
        if baseline is not None:
            baseline_method, baseline_value = baseline
            candidate_value = candidate.get(GATE_METRIC, candidate.get("wape"))
            if candidate_value is None or candidate_value > baseline_value:
                return False, (
                    f"does_not_beat_strongest_baseline:{baseline_method}={baseline_value:.4f}"
                    + (f",candidate={candidate_value:.4f}" if candidate_value is not None else "")
                )
        return True, "candidate_meets_promotion_evidence_gate"

    def _horizon_compatible(self, horizon_days: int) -> bool:
        expected = self._expected_horizon()
        if expected is None:
            return horizon_days > 0
        return horizon_days == expected

    def _expected_horizon(self) -> int | None:
        inventory = getattr(self.settings, "inventory", None)
        value = getattr(inventory, "default_lead_time_days", None)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _relative_wape_improvement(candidate: dict[str, Any], active: dict[str, Any], metric: str = GATE_METRIC) -> float | None:
        candidate_wape = candidate.get(metric, candidate.get("wape"))
        active_wape = active.get(metric, active.get("wape"))
        if candidate_wape is None or active_wape is None or float(active_wape) <= 0:
            return None
        return round((float(active_wape) - float(candidate_wape)) / float(active_wape), 6)

    @staticmethod
    def _bias_materially_worse(candidate_bias: Any, active_bias: Any) -> bool:
        if candidate_bias is None or active_bias is None:
            return False
        candidate_abs = abs(float(candidate_bias))
        active_abs = abs(float(active_bias))
        if active_abs < 1e-9:
            return candidate_abs > 0.1
        return candidate_abs > active_abs * (1.0 + BIAS_WORSENING_TOLERANCE)

    @staticmethod
    def _temporal_split(backtest: bt.BacktestResult, eval_days: int) -> dict[str, Any]:
        return {
            "method": "rolling_origin_multi_step",
            "demand_class": EVALUATED_DEMAND_CLASS,
            "eval_days": eval_days,
            "horizon_days": backtest.config.horizon,
            "n_origins": backtest.n_origins,
            "n_skus": backtest.n_skus,
            **backtest.period,
        }

    def _persist_aggregate_evaluations(
        self,
        candidate_artifact: ModelArtifact,
        active_artifact: ModelArtifact,
        result: CandidateEvaluationResult,
        scoped_metrics: dict[str, dict[str, Any]],
    ) -> None:
        assert self.session is not None
        generated_at = datetime.now(timezone.utc)
        for model_name, metrics in scoped_metrics.items():
            artifact_id = None
            if model_name == CANDIDATE_METHOD:
                artifact_id = candidate_artifact.id
            elif model_name == ACTIVE_METHOD:
                artifact_id = active_artifact.id
            self.session.add(
                ForecastEvaluation(
                    model_artifact_id=artifact_id,
                    model_name=model_name,
                    evaluation_scope="candidate_backtest",
                    metric_mae=self._decimal(metrics.get("mae")),
                    metric_rmse=self._decimal(metrics.get("rmse")),
                    metric_bias=self._decimal(metrics.get("bias")),
                    metric_wape=self._decimal(metrics.get("wape")),
                    metric_mase=self._decimal(metrics.get("mase")),
                    n_skus=int(metrics.get("n_skus") or 0),
                    n_test_points=int(metrics.get("n_test_points") or 0),
                    horizon_days=result.horizon_days,
                    generated_at=generated_at,
                )
            )
        self.session.flush()

    @staticmethod
    def _decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(round(float(value), 6)))

    @staticmethod
    def _load_model(artifact: ModelArtifact) -> Any:
        if not artifact.artifact_uri:
            raise CandidateEvaluationError("Model artifact URI is missing.")
        path = Path(artifact.artifact_uri)
        if not path.exists():
            raise CandidateEvaluationError(f"Model artifact file is missing: {path}")
        with path.open("rb") as fh:
            return pickle.load(fh)
