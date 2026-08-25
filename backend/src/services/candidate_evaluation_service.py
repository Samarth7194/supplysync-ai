"""Controlled candidate evaluation for Phase F MLOps.

This service compares a candidate LightGBM artifact against the current active
LightGBM artifact and existing statistical baselines on a temporal holdout. It
never promotes, reloads, or changes production inference.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.models import ForecastEvaluation, ModelArtifact
from evaluation import baselines
from evaluation.metrics import compute_all
from features.inference_features import build_inference_features
from services.adaptive_forecasting_service import classify_sku_demand_pattern


MODEL_NAME = "lightgbm_demand_forecast"
CANDIDATE_METHOD = "candidate_lightgbm"
ACTIVE_METHOD = "active_lightgbm"
BIAS_WORSENING_TOLERANCE = 0.20


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
        top_n: int = 20,
        persist: bool = True,
    ) -> CandidateEvaluationResult:
        if candidate_artifact.lifecycle_status != "candidate" or candidate_artifact.is_active:
            raise CandidateEvaluationError("Candidate artifact must be non-active with candidate lifecycle status.")
        if not active_artifact.is_active or active_artifact.lifecycle_status != "active":
            raise CandidateEvaluationError("Active comparison artifact is unavailable.")
        if horizon_days < 1:
            raise CandidateEvaluationError("Evaluation horizon must be positive.")

        feature_columns = list(candidate_artifact.feature_schema or [])
        if not feature_columns:
            raise CandidateEvaluationError("Candidate artifact has no feature schema.")

        candidate_model = self.model_loader(candidate_artifact)
        active_model = self.model_loader(active_artifact)
        sku_results = self._evaluate_skus(
            daily_demand=daily_demand,
            candidate_model=candidate_model,
            active_model=active_model,
            feature_columns=feature_columns,
            horizon_days=horizon_days,
            top_n=top_n,
        )
        if not sku_results:
            raise CandidateEvaluationError("No SKUs produced an evaluable temporal holdout.")

        model_names = [CANDIDATE_METHOD, ACTIVE_METHOD, "croston_sba", "moving_avg_7", "seasonal_naive_7"]
        aggregates = self._aggregate(sku_results, model_names)
        all_metrics = aggregates.get("all") or {}
        candidate_metrics = all_metrics.get(CANDIDATE_METHOD)
        active_metrics = all_metrics.get(ACTIVE_METHOD)
        if candidate_metrics is None or active_metrics is None:
            raise CandidateEvaluationError("Candidate and active LightGBM metrics are required.")

        relative = self._relative_wape_improvement(candidate_metrics, active_metrics)
        eligible, reason = self._promotion_eligibility(candidate_metrics, active_metrics, relative, horizon_days)
        temporal_split = self._temporal_split(sku_results)
        result = CandidateEvaluationResult(
            horizon_days=horizon_days,
            test_points=int(candidate_metrics.get("n_test_points") or 0),
            candidate_metrics=candidate_metrics,
            active_metrics=active_metrics,
            benchmark_metrics={k: v for k, v in all_metrics.items() if k not in {CANDIDATE_METHOD, ACTIVE_METHOD}},
            relative_wape_improvement=relative,
            promotion_eligible=eligible,
            eligibility_reason=reason,
            temporal_split=temporal_split,
        )

        candidate_artifact.training_metadata = dict(candidate_artifact.training_metadata or {})
        candidate_artifact.training_metadata["candidate_evaluation"] = result.as_dict()
        candidate_artifact.training_metrics = dict(candidate_artifact.training_metrics or {})
        candidate_artifact.training_metrics.update({
            "candidate_wape": candidate_metrics.get("wape"),
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
            self._persist_aggregate_evaluations(candidate_artifact, active_artifact, result, all_metrics)

        return result

    def _evaluate_skus(
        self,
        *,
        daily_demand: pd.DataFrame,
        candidate_model: Any,
        active_model: Any,
        feature_columns: list[str],
        horizon_days: int,
        top_n: int,
    ) -> list[dict[str, Any]]:
        required = {"StockCode", "date", "demand"}
        missing = required - set(daily_demand.columns)
        if missing:
            raise CandidateEvaluationError(f"Daily demand data missing required columns: {sorted(missing)}")

        daily = daily_demand.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        stats = daily.groupby("StockCode").agg(n=("date", "nunique"), total=("demand", "sum"))
        top_skus = stats[stats["n"] >= max(60, horizon_days + 14)].nlargest(top_n, "total").index.tolist()

        rows: list[dict[str, Any]] = []
        for sku in top_skus:
            sku_df = daily[daily["StockCode"] == sku].sort_values("date")
            series = pd.Series(sku_df["demand"].astype(float).values, index=pd.DatetimeIndex(sku_df["date"]))
            full_range = pd.date_range(series.index.min(), series.index.max(), freq="D")
            series = series.reindex(full_range, fill_value=0.0).astype(float)
            if len(series) < max(60, horizon_days + 14):
                continue
            train = series.iloc[:-horizon_days]
            test = series.iloc[-horizon_days:]
            actual = test.to_numpy(dtype=float)
            in_sample = train.to_numpy(dtype=float)

            metrics_by_model: dict[str, dict[str, Any]] = {}
            candidate_preds = self._lightgbm_predictions(candidate_model, feature_columns, train, test)
            active_preds = self._lightgbm_predictions(active_model, feature_columns, train, test)
            metrics_by_model[CANDIDATE_METHOD] = compute_all(actual, candidate_preds, in_sample=in_sample).as_dict()
            metrics_by_model[ACTIVE_METHOD] = compute_all(actual, active_preds, in_sample=in_sample).as_dict()

            for name in ("croston_sba", "moving_avg_7", "seasonal_naive_7"):
                preds = baselines.run_baseline_over_window(train, test, baselines.BASELINES[name])
                metrics_by_model[name] = compute_all(actual, preds, in_sample=in_sample).as_dict()

            rows.append({
                "sku": str(sku),
                "demand_class": classify_sku_demand_pattern(series),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_start": str(train.index.min().date()),
                "train_end": str(train.index.max().date()),
                "test_start": str(test.index.min().date()),
                "test_end": str(test.index.max().date()),
                "total_test_demand": float(actual.sum()),
                "metrics": metrics_by_model,
            })
        return rows

    @staticmethod
    def _lightgbm_predictions(model: Any, feature_columns: list[str], train: pd.Series, test: pd.Series) -> np.ndarray:
        running = train.copy()
        out = np.empty(len(test), dtype=float)
        for idx, (day, actual) in enumerate(test.items()):
            features = build_inference_features(running, feature_columns)
            if features is None:
                out[idx] = float(running.tail(7).mean()) if len(running) else 0.0
            else:
                out[idx] = max(0.0, float(model.predict(features)[0]))
            running = pd.concat([running, pd.Series([actual], index=[day])])
        return out

    @staticmethod
    def _aggregate(sku_results: list[dict[str, Any]], model_names: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
        buckets: dict[str, list[dict[str, Any]]] = {"all": list(sku_results)}
        for row in sku_results:
            buckets.setdefault(str(row["demand_class"]), []).append(row)

        output: dict[str, dict[str, dict[str, Any]]] = {}
        for bucket, rows in buckets.items():
            output[bucket] = {}
            for model_name in model_names:
                relevant = [row for row in rows if model_name in row["metrics"]]
                total_n = sum(int(row["metrics"][model_name].get("n") or 0) for row in relevant)
                if total_n <= 0:
                    continue

                def weighted_mean(key: str) -> float | None:
                    numerator = 0.0
                    denominator = 0
                    for row in relevant:
                        metric = row["metrics"][model_name]
                        value = metric.get(key)
                        n = int(metric.get("n") or 0)
                        if value is None or n <= 0:
                            continue
                        numerator += float(value) * n
                        denominator += n
                    return round(numerator / denominator, 4) if denominator else None

                sum_abs_err = 0.0
                sum_abs_actual = 0.0
                for row in relevant:
                    metric = row["metrics"][model_name]
                    if metric.get("wape") is None:
                        continue
                    total_actual = float(row["total_test_demand"])
                    sum_abs_err += float(metric["wape"]) * total_actual
                    sum_abs_actual += total_actual
                global_wape = round(sum_abs_err / sum_abs_actual, 4) if sum_abs_actual > 0 else None

                output[bucket][model_name] = {
                    "mae": weighted_mean("mae"),
                    "rmse": weighted_mean("rmse"),
                    "bias": weighted_mean("bias"),
                    "wape": global_wape,
                    "mase": weighted_mean("mase"),
                    "n_skus": len(relevant),
                    "n_test_points": total_n,
                }
        return output

    def _promotion_eligibility(
        self,
        candidate: dict[str, Any],
        active: dict[str, Any],
        relative_improvement: float | None,
        horizon_days: int,
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
    def _relative_wape_improvement(candidate: dict[str, Any], active: dict[str, Any]) -> float | None:
        candidate_wape = candidate.get("wape")
        active_wape = active.get("wape")
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
    def _temporal_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "method": "per_sku_last_n_days_holdout",
            "train_start": min(row["train_start"] for row in rows),
            "train_end": max(row["train_end"] for row in rows),
            "test_start": min(row["test_start"] for row in rows),
            "test_end": max(row["test_end"] for row in rows),
        }

    def _persist_aggregate_evaluations(
        self,
        candidate_artifact: ModelArtifact,
        active_artifact: ModelArtifact,
        result: CandidateEvaluationResult,
        all_metrics: dict[str, dict[str, Any]],
    ) -> None:
        assert self.session is not None
        generated_at = datetime.now(timezone.utc)
        for model_name, metrics in all_metrics.items():
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