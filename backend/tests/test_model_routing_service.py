from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone

from repositories.forecast_evaluation_repository import MethodPerformance
from services.model_routing_service import ModelRoutingService


@dataclass(frozen=True)
class _Settings:
    evidence_routing_enabled: bool = True
    routing_primary_metric: str = "wape"
    routing_min_evaluation_points: int = 30
    routing_min_relative_improvement: float = 0.05
    routing_evidence_lookback_days: int = 365


class _Repo:
    def __init__(self, *, sku=None, pattern=None):
        self.sku = sku or []
        self.pattern = pattern or []

    def logged_method_performance_for_sku(self, **kwargs):
        return self.sku

    def logged_method_performance_for_pattern(self, **kwargs):
        return self.pattern


def _perf(method, metric, sample=30, generated_at=None, level="pattern"):
    return MethodPerformance(
        method=method,
        metric_value=metric,
        sample_size=sample,
        evaluation_count=1,
        mean_bias=0.0,
        latest_generated_at=generated_at or datetime(2026, 8, 1, tzinfo=timezone.utc),
        horizon_days=7,
        evidence_source="logged",
        evidence_level=level,
    )


def _router(repo=None, settings=None, offline_path=None):
    return ModelRoutingService(
        settings=settings or _Settings(),
        forecast_evaluation_repository=repo,
        offline_evaluation_path=offline_path,
    )


def test_no_evidence_uses_default_method():
    decision = _router(_Repo()).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"
    assert decision.selection_source == "default"
    assert decision.fallback_used is True


def test_feature_disabled_uses_legacy_default():
    decision = _router(_Repo(), _Settings(evidence_routing_enabled=False)).select_method(
        sku_code="SKU-1",
        demand_pattern="intermittent",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "croston"
    assert decision.selection_source == "default"


def test_insufficient_sample_uses_default():
    repo = _Repo(pattern=[_perf("ml_lightgbm", 1.0, sample=29), _perf("croston", 0.5, sample=29)])

    decision = _router(repo).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"
    assert decision.selection_source == "default"


def test_candidate_meaningfully_better_is_selected():
    repo = _Repo(pattern=[_perf("ml_lightgbm", 1.0), _perf("croston", 0.8)])

    decision = _router(repo).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "croston"
    assert decision.default_method == "ml_lightgbm"
    assert decision.selection_source == "logged"
    assert decision.evidence_level == "pattern"
    assert decision.fallback_used is False


def test_slight_candidate_improvement_retains_default_for_stability():
    repo = _Repo(pattern=[_perf("ml_lightgbm", 1.0), _perf("croston", 0.98)])

    decision = _router(repo).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"
    assert "threshold" in decision.reason
    assert decision.fallback_used is False


def test_candidate_worse_retains_default():
    repo = _Repo(pattern=[_perf("ml_lightgbm", 1.0), _perf("croston", 1.2)])

    decision = _router(repo).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"


def test_tie_retains_default():
    repo = _Repo(pattern=[_perf("ml_lightgbm", 1.0), _perf("croston", 1.0)])

    decision = _router(repo).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"


def test_metric_missing_uses_default():
    repo = _Repo(pattern=[])

    decision = _router(repo, _Settings(routing_primary_metric="mase")).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"
    assert decision.selection_source == "default"


def test_stale_evidence_is_ignored_by_repository_cutoff_contract():
    stale = _perf("croston", 0.5, generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    fresh_default = _perf("ml_lightgbm", 1.0, generated_at=datetime(2026, 8, 8, tzinfo=timezone.utc))
    repo = _Repo(pattern=[fresh_default, stale])
    settings = _Settings(routing_evidence_lookback_days=1)

    decision = _router(repo, settings).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"
    assert decision.selection_source == "logged"


def test_wrong_demand_pattern_evidence_is_ignored():
    repo = _Repo(pattern=[_perf("ml_lightgbm", 1.0), _perf("croston", 0.8)])

    decision = _router(repo).select_method(
        sku_code="SKU-1",
        demand_pattern="highly_intermittent",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "conservative"
    assert decision.selection_source == "default"


def test_wrong_horizon_offline_evidence_is_ignored(tmp_path):
    path = tmp_path / "forecast_evaluation.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-01T00:00:00+00:00",
                "horizon_days": 30,
                "aggregates": {
                    "regular": {
                        "lightgbm": {"wape": 1.0, "n_test_points": 100, "bias": 0.0},
                        "croston_sba": {"wape": 0.5, "n_test_points": 100, "bias": 0.0},
                    }
                },
            }
        )
    )

    decision = _router(_Repo(), offline_path=path).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "ml_lightgbm"


def test_offline_pattern_evidence_can_bootstrap_when_horizon_matches(tmp_path):
    path = tmp_path / "forecast_evaluation.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-01T00:00:00+00:00",
                "horizon_days": 30,
                "aggregates": {
                    "regular": {
                        "lightgbm": {"wape": 1.0, "n_test_points": 100, "bias": 0.0},
                        "croston_sba": {"wape": 0.7, "n_test_points": 100, "bias": 0.0},
                    }
                },
            }
        )
    )

    decision = _router(_Repo(), offline_path=path).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=30,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "croston"
    assert decision.selection_source == "offline"
    assert decision.evidence_level == "pattern"


def test_offline_multi_horizon_sibling_evidence_is_used_when_horizon_matches(tmp_path):
    path = tmp_path / "forecast_evaluation.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-01T00:00:00+00:00",
                "horizon_days": 30,
                "aggregates": {},
            }
        )
    )
    (tmp_path / "forecast_evaluation_horizons.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-08-01T00:00:00+00:00",
                "horizons": {
                    "7": {
                        "generated_at": "2026-08-01T00:00:00+00:00",
                        "horizon_days": 7,
                        "aggregates": {
                            "regular": {
                                "lightgbm": {"wape": 1.0, "n_test_points": 100, "bias": 0.0},
                                "croston_sba": {"wape": 0.7, "n_test_points": 100, "bias": 0.0},
                            }
                        },
                    }
                },
            }
        )
    )

    decision = _router(_Repo(), offline_path=path).select_method(
        sku_code="SKU-1",
        demand_pattern="regular",
        forecast_horizon=7,
        as_of_date=date(2026, 8, 8),
    )

    assert decision.selected_method == "croston"
    assert decision.selection_source == "offline"
    assert decision.evidence_level == "pattern"
