from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base, ForecastEvaluation, ModelMonitoringSnapshot, PredictionLog, RetrainingRun
from features.schema import FEATURE_COLUMNS
from repositories.retraining_repository import RetrainingRepository
from services.historical_monitoring_replay_service import (
    HistoricalMonitoringReplayError,
    HistoricalMonitoringReplayService,
)
from services.retraining_decision_service import RetrainingDecisionService

BACKEND_DIR = Path(__file__).resolve().parents[1]
OFFLINE_EVAL_PATH = BACKEND_DIR / "data" / "forecast_evaluation.json"


# -- settings fixture (mirrors test_model_monitoring_service.py's shape,
#    extended with the retraining + inventory fields the isolation tests need)


@dataclass
class _ForecastingSettings:
    model_monitoring_enabled: bool = True
    model_monitoring_window_evaluations: int = 30
    model_monitoring_lookback_days: int = 90
    model_monitoring_min_evaluations: int = 5
    model_monitoring_wape_warning_threshold: float = 0.15
    model_monitoring_wape_degradation_threshold: float = 0.25
    model_monitoring_bias_warning_ratio: float = 0.20
    model_monitoring_degradation_consecutive_runs: int = 2
    auto_retrain_enabled: bool = False
    model_retrain_min_evaluated_forecast_days: int = 100
    model_retrain_cooldown_days: int = 14
    model_retrain_require_degraded_status: bool = True


@dataclass
class _InventorySettings:
    default_lead_time_days: int = 7


@dataclass
class _Settings:
    forecasting: _ForecastingSettings
    inventory: _InventorySettings


def _settings(**overrides) -> _Settings:
    return _Settings(forecasting=_ForecastingSettings(**overrides), inventory=_InventorySettings())


def _session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return Session()


# -- synthetic demand data ---------------------------------------------------


def _regular_series(n_days: int = 130, start: str = "2011-01-01") -> pd.Series:
    """Alternating nonzero demand — classifies as 'regular' (0% zero-share)."""
    index = pd.date_range(start=start, periods=n_days, freq="D")
    values = [8.0 if i % 2 == 0 else 12.0 for i in range(n_days)]
    return pd.Series(values, index=index)


def _sparse_series(n_days: int = 130, start: str = "2011-01-01") -> pd.Series:
    """Mostly-zero demand — classifies as 'highly_intermittent' (>80% zero-share)."""
    index = pd.date_range(start=start, periods=n_days, freq="D")
    values = [5.0 if i % 10 == 0 else 0.0 for i in range(n_days)]
    return pd.Series(values, index=index)


class _StubDataService:
    def __init__(self, series_by_sku: dict[str, pd.Series]):
        self._series = series_by_sku

    def get_demand_history(self, sku: str) -> pd.Series:
        return self._series.get(sku, pd.Series(dtype=float))

    def get_top_skus(self, n: int = 20) -> list[str]:
        return list(self._series.keys())[:n]

    def get_dataset_date_range(self):
        mins = [s.index.min() for s in self._series.values() if len(s)]
        maxs = [s.index.max() for s in self._series.values() if len(s)]
        return min(mins), max(maxs)


class _ConstantModel:
    """Predicts a fixed value regardless of input — deterministic and simple
    to hand-verify metrics against."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, features: pd.DataFrame):
        return np.full(len(features), self.value)


class _EchoLag1Model:
    """Predicts exactly `lag_1` from the feature row — makes the forecast a
    direct, checkable function of the historical input, which is what the
    no-future-leakage test needs (a constant-output model can't reveal
    leakage since its output never depends on its input)."""

    def predict(self, features: pd.DataFrame):
        return features["lag_1"].to_numpy(dtype=float)


def _spy_adaptive_forecast(monkeypatch, module):
    calls: list[dict] = []
    original = module.adaptive_forecast

    def _spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "adaptive_forecast", _spy)
    return calls


# -- root-cause-shaped fixture: dataset ends where anchors must fit inside it


def _make_service(series_by_sku, *, model=None, feature_columns=None, settings=None) -> HistoricalMonitoringReplayService:
    return HistoricalMonitoringReplayService(
        settings=settings or _settings(),
        data_service=_StubDataService(series_by_sku),
        model=model,
        feature_columns=feature_columns,
        offline_evaluation_path=str(OFFLINE_EVAL_PATH),
    )


# -- tests --------------------------------------------------------------


def test_anchor_dates_are_deterministic_and_fit_inside_the_dataset():
    dataset_max = pd.Timestamp("2011-12-09")
    anchors = HistoricalMonitoringReplayService._anchor_dates(dataset_max=dataset_max, horizon=7, num_windows=3)

    assert anchors == [
        pd.Timestamp("2011-11-18"),
        pd.Timestamp("2011-11-25"),
        pd.Timestamp("2011-12-02"),
    ]
    for anchor in anchors:
        assert anchor + pd.Timedelta(days=7) <= dataset_max


def test_no_future_leakage_forecast_depends_only_on_history_up_to_anchor():
    """Two datasets identical up to and including the anchor date, differing
    only in the target window (the future relative to that anchor), must
    produce IDENTICAL forecasts for that window — proving the future values
    never reached feature generation."""
    n_days = 100
    horizon = 7
    base = _regular_series(n_days=n_days)
    # With num_windows=1, the service anchors at dataset_max - horizon, i.e.
    # index (n_days - 1) - horizon. Mutating strictly after that index touches
    # only the target window, never the history the forecast is built from.
    anchor_index = (n_days - 1) - horizon

    series_a = base.copy()
    series_b = base.copy()
    series_b.iloc[anchor_index + 1 :] = series_b.iloc[anchor_index + 1 :] + 1000.0

    model = _EchoLag1Model()
    service_a = _make_service({"SKU-1": series_a}, model=model, feature_columns=FEATURE_COLUMNS)
    service_b = _make_service({"SKU-1": series_b}, model=model, feature_columns=FEATURE_COLUMNS)

    result_a = service_a.run(horizon_days=horizon, sku_limit=1, num_windows=1, min_history_days=60)
    result_b = service_b.run(horizon_days=horizon, sku_limit=1, num_windows=1, min_history_days=60)

    predicted_a = result_a.windows[0].sku_results[0].predicted
    predicted_b = result_b.windows[0].sku_results[0].predicted
    assert predicted_a == predicted_b, "forecast changed when only future-of-anchor data changed — leakage"

    # The actuals legitimately differ (that's the point of the mutation).
    actual_a = result_a.windows[0].sku_results[0].actual
    actual_b = result_b.windows[0].sku_results[0].actual
    assert actual_a != actual_b


def test_demand_classification_uses_only_pre_anchor_history():
    """classify_sku_demand_pattern must run on the anchor-truncated history,
    never on the full series — a future segment with a very different demand
    shape must not change how the SKU is classified (and therefore routed)
    at the anchor."""
    n_days = 100
    horizon = 7
    anchor_index = (n_days - 1) - horizon

    # Regular (no zeros) up to and including the anchor; the tail (target
    # window and beyond) becomes highly intermittent (mostly zero). If the
    # classifier ever saw the full series, it would flip the classification
    # away from "regular" because of the future zero-heavy segment.
    index = pd.date_range("2011-01-01", periods=n_days, freq="D")
    values = [8.0 if i % 2 == 0 else 12.0 for i in range(n_days)]
    for i in range(anchor_index + 1, n_days):
        values[i] = 0.0
    series = pd.Series(values, index=index)

    import services.historical_monitoring_replay_service as replay_module

    seen_patterns: list[str] = []
    original_classify = replay_module.classify_sku_demand_pattern

    def _spy_classify(demand_series):
        pattern = original_classify(demand_series)
        seen_patterns.append(pattern)
        # The series handed to the classifier must never extend past the
        # anchor date used for this window.
        assert demand_series.index.max() <= index[anchor_index]
        return pattern

    import pytest as _pytest  # local import to avoid polluting module namespace

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(replay_module, "classify_sku_demand_pattern", _spy_classify)
        service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)
        service.run(horizon_days=horizon, sku_limit=1, num_windows=1, min_history_days=60)

    assert seen_patterns == ["regular"], (
        "classification leaked future demand: the future zero-heavy tail "
        "would have made this 'highly_intermittent' if it were visible"
    )


def test_no_future_leakage_adaptive_forecast_never_receives_post_anchor_dates(monkeypatch):
    import services.historical_monitoring_replay_service as replay_module

    calls = _spy_adaptive_forecast(monkeypatch, replay_module)
    series = _regular_series(n_days=130)
    service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)

    result = service.run(horizon_days=7, sku_limit=1, num_windows=3, min_history_days=60)

    assert len(calls) == 3  # one call per window (single SKU)
    for call, window in zip(calls, result.windows):
        anchor = pd.Timestamp(window.anchor_date)
        demand_series = call["demand_series"]
        assert demand_series.index.max() <= anchor


def test_metrics_are_computed_correctly_against_hand_derived_values():
    series = _regular_series(n_days=100)
    service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)

    result = service.run(horizon_days=7, sku_limit=1, num_windows=1, min_history_days=60)
    sku_result = result.windows[0].sku_results[0]

    actual = np.array(sku_result.actual)
    predicted = np.array(sku_result.predicted)
    assert np.all(predicted == 10.0)

    expected_wape = float(np.abs(actual - predicted).sum() / np.abs(actual).sum())
    expected_mae = float(np.abs(actual - predicted).mean())
    expected_bias = float(np.mean(predicted - actual))

    # sku_result.metrics is the rounded-to-4-decimal dict from ForecastMetrics.as_dict().
    assert sku_result.metrics["wape"] == pytest.approx(expected_wape, abs=1e-3)
    assert sku_result.metrics["mae"] == pytest.approx(expected_mae, abs=1e-3)
    assert sku_result.metrics["bias"] == pytest.approx(expected_bias, abs=1e-3)

    # Top-level aggregate must match the single-SKU/single-window case exactly.
    assert result.metric_wape == pytest.approx(expected_wape, abs=1e-6)
    assert result.metric_mae == pytest.approx(expected_mae, abs=1e-6)


def test_provenance_is_always_historical_replay():
    series = _regular_series(n_days=100)
    service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)

    result = service.run(horizon_days=7, sku_limit=1, num_windows=1, min_history_days=60)

    assert result.provenance == "historical_replay"
    assert result.as_dict()["provenance"] == "historical_replay"


def test_replay_is_deterministic_across_repeated_runs():
    series = _regular_series(n_days=130)
    service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)

    first = service.run(horizon_days=7, sku_limit=1, num_windows=3, min_history_days=60).as_dict()
    second = service.run(horizon_days=7, sku_limit=1, num_windows=3, min_history_days=60).as_dict()

    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second


def test_hybrid_methods_are_represented_honestly_and_status_scoped_to_lightgbm():
    regular = _regular_series(n_days=130)
    sparse = _sparse_series(n_days=130)
    service = _make_service(
        {"REG-1": regular, "SPARSE-1": sparse},
        model=_ConstantModel(10.0),
        feature_columns=FEATURE_COLUMNS,
    )

    result = service.run(horizon_days=7, sku_limit=2, num_windows=1, min_history_days=60)

    assert "ml_lightgbm" in result.method_breakdown
    assert "conservative" in result.method_breakdown
    assert result.method_breakdown["ml_lightgbm"]["sku_count"] == 1
    assert result.method_breakdown["conservative"]["sku_count"] == 1

    # Artifact-level status/evaluation_count reflects ONLY the lightgbm-routed
    # SKU — the conservative SKU must not be folded into artifact health.
    lightgbm_only_points = result.method_breakdown["ml_lightgbm"]["evaluation_count"]
    assert result.evaluation_count == lightgbm_only_points


def test_insufficient_evidence_when_too_few_lightgbm_evaluations():
    series = _regular_series(n_days=100)
    settings = _settings(model_monitoring_min_evaluations=1000)  # unreachable on purpose
    service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS, settings=settings)

    result = service.run(horizon_days=7, sku_limit=1, num_windows=1, min_history_days=60)

    assert result.status == "insufficient_evidence"


def test_run_rejects_invalid_horizon_and_window_count():
    series = _regular_series(n_days=100)
    service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)

    with pytest.raises(HistoricalMonitoringReplayError):
        service.run(horizon_days=0)
    with pytest.raises(HistoricalMonitoringReplayError):
        service.run(num_windows=0)


def test_run_rejects_empty_sku_universe():
    service = _make_service({}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)
    with pytest.raises(HistoricalMonitoringReplayError):
        service.run()


# -- isolation from live evidence / retraining -------------------------------


def test_replay_never_writes_to_live_evidence_tables():
    session = _session()
    series = _regular_series(n_days=130)
    service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS)

    service.run(horizon_days=7, sku_limit=1, num_windows=3, min_history_days=60)

    assert session.scalars(select(PredictionLog)).all() == []
    assert session.scalars(select(ForecastEvaluation)).all() == []
    assert session.scalars(select(ModelMonitoringSnapshot)).all() == []
    assert session.scalars(select(RetrainingRun)).all() == []


def test_replay_does_not_change_retraining_recommendation():
    session = _session()
    settings = _settings()
    retraining_service = RetrainingDecisionService(repository=RetrainingRepository(session), settings=settings)

    before = retraining_service.evaluate(persist_recommendation=False)

    series = _regular_series(n_days=130)
    replay_service = _make_service({"SKU-1": series}, model=_ConstantModel(10.0), feature_columns=FEATURE_COLUMNS, settings=settings)
    replay_service.run(horizon_days=7, sku_limit=1, num_windows=3, min_history_days=60)

    after = retraining_service.evaluate(persist_recommendation=False)

    assert before.recommended is False
    assert before.reason == "model_unavailable"
    assert after.recommended == before.recommended
    assert after.reason == before.reason


def test_replay_module_has_no_training_promotion_or_db_write_capability():
    """Structural guard: the replay service must be unable to train a model,
    promote/rollback an artifact, or open its own DB session — the isolation
    guarantee should hold even if a future change tried to add persistence
    without thinking it through."""
    source = Path(replay_module_path()).read_text()
    forbidden = [
        "CandidateTrainingService",
        "ModelPromotionService",
        "ModelArtifactRepository",
        "ModelMonitoringRepository",
        "RetrainingRepository",
        "SessionLocal",
        "session.add(",
        "session.commit(",
        "from sqlalchemy.orm import Session",
    ]
    for token in forbidden:
        assert token not in source, f"unexpected coupling found: {token}"


def replay_module_path() -> Path:
    import services.historical_monitoring_replay_service as module

    return Path(module.__file__)


def test_mlops_cycle_service_has_no_historical_replay_coupling():
    import services.mlops_cycle_service as module

    source = Path(module.__file__).read_text()
    assert "historical_replay" not in source
    assert "HistoricalMonitoringReplayService" not in source
