"""Historical monitoring replay.

The production dataset is a static historical retail extract (2009-12-01
through 2011-12-09) with no connected ERP/POS actual-demand stream. Live
predictions logged "today" therefore cannot mature into real
``forecast_evaluations`` — their target windows extend past the dataset's
end date, so actual demand is never available for them.

This service demonstrates the same evaluation/monitoring lifecycle honestly
by replaying it against held-out historical windows: pick an anchor date T
inside the dataset, forecast using only demand history <= T (the same hybrid
routing and forecasting code the live system uses), then compare against the
*already-recorded* actual demand for T+1..T+H.

This is NOT live production monitoring. Every result this service returns is
tagged ``provenance = "historical_replay"``. It never writes to
``prediction_logs``, ``forecast_evaluations``, ``model_monitoring_snapshots``,
or ``retraining_runs`` — it only reads the processed parquet (via
``DataService``) and an in-memory model, and returns a plain result object.
``RetrainingDecisionService``, ``ModelMonitoringService.create_snapshot``, and
``ModelRoutingService`` are therefore structurally unaffected by this module;
nothing here can leak into live retraining/degradation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import compute_all
from services.adaptive_forecasting_service import adaptive_forecast, classify_sku_demand_pattern
from services.model_monitoring_service import Baseline, ModelMonitoringService, MonitoringMetrics

HISTORICAL_REPLAY_PROVENANCE = "historical_replay"
ARTIFACT_SCOPED_METHOD = "ml_lightgbm"
DEFAULT_MIN_HISTORY_DAYS = 60
DEFAULT_SKU_LIMIT = 60
DEFAULT_NUM_WINDOWS = 3


class HistoricalMonitoringReplayError(Exception):
    """Raised when a historical replay cannot be produced safely."""


@dataclass(frozen=True)
class SkuWindowResult:
    sku: str
    demand_class: str
    forecast_method: str
    n_test_points: int
    total_actual_demand: float
    actual: tuple[float, ...]
    predicted: tuple[float, ...]
    metrics: dict[str, float | None]


@dataclass(frozen=True)
class ReplayWindowResult:
    window_index: int
    anchor_date: date
    target_start: date
    target_end: date
    sku_results: list[SkuWindowResult]
    status: str
    degradation_reason: str
    degradation_message: str
    consecutive_degradation_count: int
    metric_wape: float | None
    evaluation_count: int


@dataclass(frozen=True)
class HistoricalReplayResult:
    provenance: str
    generated_at: datetime
    model_name: str
    model_artifact_id: int | None
    model_version: str | None
    horizon_days: int
    sku_count: int
    window_count: int
    windows: list[ReplayWindowResult]
    status: str
    degradation_reason: str
    degradation_message: str
    evaluation_count: int
    metric_wape: float | None
    metric_mae: float | None
    metric_rmse: float | None
    metric_bias: float | None
    metric_mase: float | None
    residual_mean: float | None
    residual_std: float | None
    baseline_wape: float | None
    baseline_provenance: str
    wape_relative_change: float | None
    bias_ratio: float | None
    consecutive_degradation_count: int
    historical_period_start: date | None
    historical_period_end: date | None
    method_breakdown: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "generated_at": self.generated_at.isoformat(),
            "model_name": self.model_name,
            "model_artifact_id": self.model_artifact_id,
            "model_version": self.model_version,
            "horizon_days": self.horizon_days,
            "sku_count": self.sku_count,
            "window_count": self.window_count,
            "status": self.status,
            "degradation_reason": self.degradation_reason,
            "degradation_message": self.degradation_message,
            "evaluation_count": self.evaluation_count,
            "metric_wape": self.metric_wape,
            "metric_mae": self.metric_mae,
            "metric_rmse": self.metric_rmse,
            "metric_bias": self.metric_bias,
            "metric_mase": self.metric_mase,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "baseline_wape": self.baseline_wape,
            "baseline_provenance": self.baseline_provenance,
            "wape_relative_change": self.wape_relative_change,
            "bias_ratio": self.bias_ratio,
            "consecutive_degradation_count": self.consecutive_degradation_count,
            "historical_period": {
                "start": self.historical_period_start.isoformat() if self.historical_period_start else None,
                "end": self.historical_period_end.isoformat() if self.historical_period_end else None,
            },
            "method_breakdown": self.method_breakdown,
            "windows": [
                {
                    "window_index": w.window_index,
                    "anchor_date": w.anchor_date.isoformat(),
                    "target_start": w.target_start.isoformat(),
                    "target_end": w.target_end.isoformat(),
                    "status": w.status,
                    "degradation_reason": w.degradation_reason,
                    "consecutive_degradation_count": w.consecutive_degradation_count,
                    "metric_wape": w.metric_wape,
                    "evaluation_count": w.evaluation_count,
                    "sku_count": len(w.sku_results),
                }
                for w in self.windows
            ],
        }


class HistoricalMonitoringReplayService:
    """Replays the evaluation/monitoring lifecycle over historical windows.

    Read-only with respect to the database: this service never receives or
    uses a SQLAlchemy session. It only needs a loaded forecasting model (or
    ``None``, in which case regular-demand SKUs fall back the same way
    ``adaptive_forecast`` does in production), the feature-column schema, and
    a ``DataService``-shaped source of historical demand.
    """

    def __init__(
        self,
        *,
        settings: Any,
        data_service: Any,
        model: Any | None = None,
        feature_columns: list[str] | None = None,
        model_name: str = "lightgbm_demand_forecast",
        model_artifact_id: int | None = None,
        model_version: str | None = None,
        offline_evaluation_path: str | None = None,
    ):
        self.settings = settings
        self.data_service = data_service
        self.model = model
        self.feature_columns = feature_columns
        self.model_name = model_name
        self.model_artifact_id = model_artifact_id
        self.model_version = model_version
        # A throwaway ModelMonitoringService instance, used only to reuse its
        # pure classification/baseline helpers. It is never given a real
        # repository and `create_snapshot`/`current_snapshot` are never
        # called, so no live monitoring behavior is touched.
        self._classifier = ModelMonitoringService(
            repository=None,
            settings=settings,
            offline_evaluation_path=offline_evaluation_path,
        )

    def run(
        self,
        *,
        horizon_days: int | None = None,
        sku_limit: int = DEFAULT_SKU_LIMIT,
        num_windows: int = DEFAULT_NUM_WINDOWS,
        min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    ) -> HistoricalReplayResult:
        horizon = int(horizon_days) if horizon_days is not None else int(getattr(self.settings.inventory, "default_lead_time_days", 7))
        if horizon < 1:
            raise HistoricalMonitoringReplayError("horizon_days must be >= 1.")
        if num_windows < 1:
            raise HistoricalMonitoringReplayError("num_windows must be >= 1.")

        candidate_skus = self._candidate_skus(sku_limit=sku_limit)
        if not candidate_skus:
            raise HistoricalMonitoringReplayError("No SKUs available in the processed dataset.")

        dataset_min, dataset_max = self.data_service.get_dataset_date_range()
        dataset_min = pd.Timestamp(dataset_min)
        dataset_max = pd.Timestamp(dataset_max)

        anchors = self._anchor_dates(dataset_max=dataset_max, horizon=horizon, num_windows=num_windows)

        windows: list[ReplayWindowResult] = []
        window_metrics: list[MonitoringMetrics] = []
        previous_state: SimpleNamespace | None = None
        baseline = self._baseline(horizon_days=horizon)

        for index, anchor in enumerate(anchors):
            target_start = anchor + pd.Timedelta(days=1)
            target_end = anchor + pd.Timedelta(days=horizon)
            sku_results = self._evaluate_window(
                skus=candidate_skus,
                anchor=anchor,
                target_start=target_start,
                target_end=target_end,
                horizon=horizon,
                min_history_days=min_history_days,
            )
            # Each window's own evaluation count is what matters here — this
            # mirrors live monitoring, where one snapshot's sufficiency comes
            # from its own rolling window, not from pooling across snapshots.
            artifact_scoped = [r for r in sku_results if r.forecast_method == ARTIFACT_SCOPED_METHOD]
            metrics = self._aggregate_metrics(artifact_scoped)
            state = self._classifier._performance_state(
                metrics=metrics,
                baseline=baseline,
                evaluation_count=len(artifact_scoped),
                previous_snapshots=[previous_state] if previous_state is not None else [],
            )
            windows.append(
                ReplayWindowResult(
                    window_index=index,
                    anchor_date=anchor.date(),
                    target_start=target_start.date(),
                    target_end=target_end.date(),
                    sku_results=sku_results,
                    status=state.status,
                    degradation_reason=state.reason,
                    degradation_message=state.message,
                    consecutive_degradation_count=state.consecutive_degradation_count,
                    metric_wape=metrics.wape,
                    evaluation_count=len(artifact_scoped),
                )
            )
            window_metrics.append(metrics)
            previous_state = SimpleNamespace(
                degradation_reason=state.reason,
                consecutive_degradation_count=state.consecutive_degradation_count,
            )

        # The reported top-level status is the most recent window's already
        # -computed classification — recomputing it here would be redundant
        # and risks silently diverging from the per-window result above.
        latest_window = windows[-1]
        latest_metrics = window_metrics[-1]
        bias_ratio = self._classifier._bias_ratio(latest_metrics)
        method_breakdown = self._method_breakdown(windows)
        all_sku_codes = sorted({r.sku for w in windows for r in w.sku_results})

        return HistoricalReplayResult(
            provenance=HISTORICAL_REPLAY_PROVENANCE,
            generated_at=datetime.now(timezone.utc),
            model_name=self.model_name,
            model_artifact_id=self.model_artifact_id,
            model_version=self.model_version,
            horizon_days=horizon,
            sku_count=len(all_sku_codes),
            window_count=len(windows),
            windows=windows,
            status=latest_window.status,
            degradation_reason=latest_window.degradation_reason,
            degradation_message=latest_window.degradation_message,
            evaluation_count=latest_window.evaluation_count,
            metric_wape=latest_metrics.wape,
            metric_mae=latest_metrics.mae,
            metric_rmse=latest_metrics.rmse,
            metric_bias=latest_metrics.bias,
            metric_mase=latest_metrics.mase,
            residual_mean=latest_metrics.residual_mean,
            residual_std=latest_metrics.residual_std,
            baseline_wape=baseline.wape,
            baseline_provenance=baseline.provenance,
            wape_relative_change=self._classifier._wape_relative_change(latest_metrics.wape, baseline.wape),
            bias_ratio=bias_ratio,
            consecutive_degradation_count=latest_window.consecutive_degradation_count,
            historical_period_start=windows[0].target_start if windows else None,
            historical_period_end=windows[-1].target_end if windows else None,
            method_breakdown=method_breakdown,
        )

    # -- window construction --------------------------------------------

    @staticmethod
    def _anchor_dates(*, dataset_max: pd.Timestamp, horizon: int, num_windows: int) -> list[pd.Timestamp]:
        """Deterministic, reproducible anchors: non-overlapping windows
        walking backward from the dataset's end, oldest first.

        anchor_0 = dataset_max - horizon (the most recent window whose target
        still fully fits inside the dataset), anchor_1 = anchor_0 - horizon,
        and so on. No randomness anywhere.
        """
        anchors = [dataset_max - pd.Timedelta(days=horizon * (i + 1)) for i in range(num_windows)]
        anchors.reverse()  # oldest first, so classification state threads forward in time
        return anchors

    def _candidate_skus(self, *, sku_limit: int) -> list[str]:
        return self.data_service.get_top_skus(n=sku_limit)

    def _evaluate_window(
        self,
        *,
        skus: list[str],
        anchor: pd.Timestamp,
        target_start: pd.Timestamp,
        target_end: pd.Timestamp,
        horizon: int,
        min_history_days: int,
    ) -> list[SkuWindowResult]:
        results: list[SkuWindowResult] = []
        for sku in skus:
            series = self.data_service.get_demand_history(sku)
            if series.empty or not isinstance(series.index, pd.DatetimeIndex):
                continue

            # History strictly on-or-before the anchor. Slicing at `anchor`
            # cannot include any date after it, so the model never sees
            # target-window demand as input — no leakage by construction.
            history = series.loc[:anchor]
            if len(history) < min_history_days:
                continue

            if target_start < series.index.min() or target_end > series.index.max():
                continue
            actual_window = series.loc[target_start:target_end]
            if len(actual_window) != horizon:
                continue

            demand_pattern = classify_sku_demand_pattern(history)
            forecast, method = adaptive_forecast(
                sku=sku,
                demand_series=history,
                horizon=horizon,
                model=self.model,
                feature_columns=self.feature_columns,
                routing_service=None,
                include_routing=False,
            )
            predicted = np.asarray(forecast[:horizon], dtype=float)
            actual = actual_window.to_numpy(dtype=float)
            in_sample = history.to_numpy(dtype=float)
            metrics = compute_all(actual, predicted, in_sample=in_sample, seasonality=1)

            results.append(
                SkuWindowResult(
                    sku=sku,
                    demand_class=demand_pattern,
                    forecast_method=method,
                    n_test_points=metrics.n,
                    total_actual_demand=float(actual.sum()),
                    actual=tuple(float(v) for v in actual),
                    predicted=tuple(float(v) for v in predicted),
                    metrics=metrics.as_dict(),
                )
            )
        return results

    # -- aggregation & classification (reuses ModelMonitoringService math) --

    @staticmethod
    def _aggregate_metrics(results: list[SkuWindowResult]) -> MonitoringMetrics:
        if not results:
            return MonitoringMetrics(None, None, None, None, None, None, None, None)

        actual_all = np.concatenate([np.asarray(r.actual, dtype=float) for r in results])
        predicted_all = np.concatenate([np.asarray(r.predicted, dtype=float) for r in results])
        residuals = actual_all - predicted_all
        absolute_error = np.abs(residuals)
        actual_demand = np.abs(actual_all).sum()

        mase_pairs = [(r.metrics["mase"], r.n_test_points) for r in results if r.metrics.get("mase") is not None]
        mase = ModelMonitoringService._weighted_pairs(mase_pairs)

        return MonitoringMetrics(
            wape=float(absolute_error.sum() / actual_demand) if actual_demand > 0 else None,
            mae=float(absolute_error.mean()) if absolute_error.size else None,
            rmse=float(np.sqrt(np.mean(residuals**2))) if residuals.size else None,
            bias=float(np.mean(predicted_all - actual_all)) if actual_all.size else None,
            mase=mase,
            residual_mean=float(np.mean(residuals)) if residuals.size else None,
            residual_std=float(np.std(residuals, ddof=1)) if residuals.size >= 2 else None,
            mean_actual_demand=float(np.mean(np.abs(actual_all))) if actual_all.size else None,
        )

    def _baseline(self, *, horizon_days: int) -> Baseline:
        pseudo_evaluations = [SimpleNamespace(horizon_days=horizon_days)]
        wape = self._classifier._offline_baseline(pseudo_evaluations)
        if wape is not None:
            return Baseline(wape, "offline_backtest")
        return Baseline(None, "unavailable")

    @staticmethod
    def _method_breakdown(windows: list[ReplayWindowResult]) -> dict[str, dict[str, Any]]:
        """Honest per-method summary across all windows (Croston/conservative
        included) — informational only, never fed into the artifact-level
        status, which is scoped to the LightGBM artifact being monitored."""
        by_method: dict[str, list[SkuWindowResult]] = {}
        for window in windows:
            for result in window.sku_results:
                by_method.setdefault(result.forecast_method, []).append(result)

        breakdown: dict[str, dict[str, Any]] = {}
        for method, rows in by_method.items():
            actual_all = np.concatenate([np.asarray(r.actual, dtype=float) for r in rows])
            predicted_all = np.concatenate([np.asarray(r.predicted, dtype=float) for r in rows])
            abs_err = np.abs(actual_all - predicted_all)
            actual_sum = np.abs(actual_all).sum()
            breakdown[method] = {
                "sku_count": len({r.sku for r in rows}),
                "evaluation_count": len(rows),
                "wape": float(abs_err.sum() / actual_sum) if actual_sum > 0 else None,
            }
        return breakdown
