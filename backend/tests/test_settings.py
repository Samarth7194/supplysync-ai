"""Tests for typed backend settings.

The settings module is the single runtime configuration entry point for the
FastAPI app. These tests keep env parsing predictable as more production
settings are added.
"""

from pathlib import Path


def test_settings_defaults(monkeypatch):
    from config.settings import BACKEND_DIR, load_settings

    for name in (
        "MODEL_PATH",
        "DATABASE_URL",
        "ALLOWED_ORIGINS",
        "LEAD_TIME_DAYS",
        "SERVICE_LEVEL",
        "HOLDING_COST",
        "STOCKOUT_COST",
        "LOG_LEVEL",
        "LOG_JSON",
        "EVIDENCE_ROUTING_ENABLED",
        "ROUTING_PRIMARY_METRIC",
        "ROUTING_MIN_EVALUATION_POINTS",
        "ROUTING_MIN_RELATIVE_IMPROVEMENT",
        "ROUTING_EVIDENCE_LOOKBACK_DAYS",
        "MODEL_MONITORING_ENABLED",
        "MODEL_MONITORING_WINDOW_EVALUATIONS",
        "MODEL_MONITORING_LOOKBACK_DAYS",
        "MODEL_MONITORING_MIN_EVALUATIONS",
        "MODEL_MONITORING_WAPE_WARNING_THRESHOLD",
        "MODEL_MONITORING_WAPE_DEGRADATION_THRESHOLD",
        "MODEL_MONITORING_BIAS_WARNING_RATIO",
        "MODEL_MONITORING_DEGRADATION_CONSECUTIVE_RUNS",
        "AUTO_RETRAIN_ENABLED",
        "MODEL_RETRAIN_MIN_EVALUATED_FORECAST_DAYS",
        "MODEL_RETRAIN_COOLDOWN_DAYS",
        "MODEL_RETRAIN_REQUIRE_DEGRADED_STATUS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.forecasting.model_path == str(BACKEND_DIR / "saved_models")
    assert settings.forecasting.evidence_routing_enabled is False
    assert settings.forecasting.routing_primary_metric == "wape"
    assert settings.forecasting.routing_min_evaluation_points == 30
    assert settings.forecasting.routing_min_relative_improvement == 0.05
    assert settings.forecasting.routing_evidence_lookback_days == 365
    assert settings.forecasting.model_monitoring_enabled is True
    assert settings.forecasting.model_monitoring_window_evaluations == 30
    assert settings.forecasting.model_monitoring_lookback_days == 90
    assert settings.forecasting.model_monitoring_min_evaluations == 30
    assert settings.forecasting.model_monitoring_wape_warning_threshold == 0.15
    assert settings.forecasting.model_monitoring_wape_degradation_threshold == 0.25
    assert settings.forecasting.model_monitoring_bias_warning_ratio == 0.20
    assert settings.forecasting.model_monitoring_degradation_consecutive_runs == 2
    assert settings.forecasting.auto_retrain_enabled is False
    assert settings.forecasting.model_retrain_min_evaluated_forecast_days == 100
    assert settings.forecasting.model_retrain_cooldown_days == 14
    assert settings.forecasting.model_retrain_require_degraded_status is True
    assert settings.database.database_url == f"sqlite:///{(BACKEND_DIR / 'data' / 'supplysync.db').as_posix()}"
    assert settings.app.allowed_origins == ["http://localhost:3000"]
    assert settings.inventory.default_lead_time_days == 7
    assert settings.inventory.default_service_level == 0.95
    assert settings.inventory.holding_cost_per_unit == 0.2
    assert settings.inventory.stockout_cost_per_unit == 1.0
    assert settings.logging.level == "INFO"
    assert settings.logging.json_logs is False


def test_settings_parse_env_overrides(monkeypatch, tmp_path):
    from config.settings import load_settings

    model_dir = tmp_path / "models"
    monkeypatch.setenv("MODEL_PATH", str(model_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://supplysync:supplysync@localhost:5432/supplysync")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000, https://demo.example.com")
    monkeypatch.setenv("LEAD_TIME_DAYS", "14")
    monkeypatch.setenv("SERVICE_LEVEL", "0.9")
    monkeypatch.setenv("HOLDING_COST", "0.4")
    monkeypatch.setenv("STOCKOUT_COST", "5")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_JSON", "true")
    monkeypatch.setenv("EVIDENCE_ROUTING_ENABLED", "false")
    monkeypatch.setenv("ROUTING_PRIMARY_METRIC", "mase")
    monkeypatch.setenv("ROUTING_MIN_EVALUATION_POINTS", "50")
    monkeypatch.setenv("ROUTING_MIN_RELATIVE_IMPROVEMENT", "0.15")
    monkeypatch.setenv("ROUTING_EVIDENCE_LOOKBACK_DAYS", "90")
    monkeypatch.setenv("MODEL_MONITORING_ENABLED", "false")
    monkeypatch.setenv("MODEL_MONITORING_WINDOW_EVALUATIONS", "40")
    monkeypatch.setenv("MODEL_MONITORING_LOOKBACK_DAYS", "120")
    monkeypatch.setenv("MODEL_MONITORING_MIN_EVALUATIONS", "20")
    monkeypatch.setenv("MODEL_MONITORING_WAPE_WARNING_THRESHOLD", "0.10")
    monkeypatch.setenv("MODEL_MONITORING_WAPE_DEGRADATION_THRESHOLD", "0.30")
    monkeypatch.setenv("MODEL_MONITORING_BIAS_WARNING_RATIO", "0.25")
    monkeypatch.setenv("MODEL_MONITORING_DEGRADATION_CONSECUTIVE_RUNS", "3")
    monkeypatch.setenv("AUTO_RETRAIN_ENABLED", "false")
    monkeypatch.setenv("MODEL_RETRAIN_MIN_EVALUATED_FORECAST_DAYS", "140")
    monkeypatch.setenv("MODEL_RETRAIN_COOLDOWN_DAYS", "7")
    monkeypatch.setenv("MODEL_RETRAIN_REQUIRE_DEGRADED_STATUS", "true")

    settings = load_settings()

    assert settings.forecasting.model_path == str(model_dir)
    assert settings.database.database_url == "postgresql+psycopg2://supplysync:supplysync@localhost:5432/supplysync"
    assert settings.app.allowed_origins == ["http://localhost:3000", "https://demo.example.com"]
    assert settings.inventory.default_lead_time_days == 14
    assert settings.inventory.default_service_level == 0.9
    assert settings.inventory.holding_cost_per_unit == 0.4
    assert settings.inventory.stockout_cost_per_unit == 5.0
    assert settings.logging.level == "DEBUG"
    assert settings.logging.json_logs is True
    assert settings.forecasting.evidence_routing_enabled is False
    assert settings.forecasting.routing_primary_metric == "mase"
    assert settings.forecasting.routing_min_evaluation_points == 50
    assert settings.forecasting.routing_min_relative_improvement == 0.15
    assert settings.forecasting.routing_evidence_lookback_days == 90
    assert settings.forecasting.model_monitoring_enabled is False
    assert settings.forecasting.model_monitoring_window_evaluations == 40
    assert settings.forecasting.model_monitoring_lookback_days == 120
    assert settings.forecasting.model_monitoring_min_evaluations == 20
    assert settings.forecasting.model_monitoring_wape_warning_threshold == 0.10
    assert settings.forecasting.model_monitoring_wape_degradation_threshold == 0.30
    assert settings.forecasting.model_monitoring_bias_warning_ratio == 0.25
    assert settings.forecasting.model_monitoring_degradation_consecutive_runs == 3
    assert settings.forecasting.auto_retrain_enabled is False
    assert settings.forecasting.model_retrain_min_evaluated_forecast_days == 140
    assert settings.forecasting.model_retrain_cooldown_days == 7
    assert settings.forecasting.model_retrain_require_degraded_status is True


def test_settings_resolve_repo_root_style_backend_paths(monkeypatch):
    from config.settings import PROJECT_ROOT, load_settings

    monkeypatch.setenv("MODEL_PATH", "backend/saved_models")

    settings = load_settings()

    assert Path(settings.forecasting.model_path) == PROJECT_ROOT / "backend" / "saved_models"


def test_settings_invalid_numbers_fail_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("LEAD_TIME_DAYS", "not-int")

    with pytest.raises(ValueError, match="LEAD_TIME_DAYS must be an integer"):
        load_settings()


def test_settings_invalid_inventory_ranges_fail_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("SERVICE_LEVEL", "7")

    with pytest.raises(ValueError, match="SERVICE_LEVEL must satisfy"):
        load_settings()


def test_settings_invalid_logging_values_fail_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("LOG_LEVEL", "LOUD")

    with pytest.raises(ValueError, match="LOG_LEVEL must be one of"):
        load_settings()


def test_settings_invalid_routing_metric_fails_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("ROUTING_PRIMARY_METRIC", "mape")

    with pytest.raises(ValueError, match="ROUTING_PRIMARY_METRIC"):
        load_settings()


def test_settings_invalid_routing_threshold_fails_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("ROUTING_MIN_RELATIVE_IMPROVEMENT", "2")

    with pytest.raises(ValueError, match="ROUTING_MIN_RELATIVE_IMPROVEMENT"):
        load_settings()


def test_settings_invalid_monitoring_threshold_order_fails_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("MODEL_MONITORING_WAPE_WARNING_THRESHOLD", "0.30")
    monkeypatch.setenv("MODEL_MONITORING_WAPE_DEGRADATION_THRESHOLD", "0.25")

    with pytest.raises(ValueError, match="MODEL_MONITORING_WAPE_WARNING_THRESHOLD"):
        load_settings()


def test_settings_invalid_monitoring_consecutive_runs_fails_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("MODEL_MONITORING_DEGRADATION_CONSECUTIVE_RUNS", "0")

    with pytest.raises(ValueError, match="MODEL_MONITORING_DEGRADATION_CONSECUTIVE_RUNS"):
        load_settings()


def test_settings_invalid_retraining_min_forecast_days_fails_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("MODEL_RETRAIN_MIN_EVALUATED_FORECAST_DAYS", "0")

    with pytest.raises(ValueError, match="MODEL_RETRAIN_MIN_EVALUATED_FORECAST_DAYS"):
        load_settings()


def test_settings_invalid_retraining_cooldown_fails_fast(monkeypatch):
    import pytest
    from config.settings import load_settings

    monkeypatch.setenv("MODEL_RETRAIN_COOLDOWN_DAYS", "-1")

    with pytest.raises(ValueError, match="MODEL_RETRAIN_COOLDOWN_DAYS"):
        load_settings()
