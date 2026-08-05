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
        "ANALYSES_DB_PATH",
        "DATABASE_URL",
        "ALLOWED_ORIGINS",
        "LEAD_TIME_DAYS",
        "SERVICE_LEVEL",
        "HOLDING_COST",
        "STOCKOUT_COST",
        "LOG_LEVEL",
        "LOG_JSON",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.forecasting.model_path == str(BACKEND_DIR / "saved_models")
    assert settings.database.analyses_db_path == str(BACKEND_DIR / "data" / "analyses.sqlite")
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
    db_path = tmp_path / "analyses.sqlite"
    monkeypatch.setenv("MODEL_PATH", str(model_dir))
    monkeypatch.setenv("ANALYSES_DB_PATH", str(db_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://supplysync:supplysync@localhost:5432/supplysync")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000, https://demo.example.com")
    monkeypatch.setenv("LEAD_TIME_DAYS", "14")
    monkeypatch.setenv("SERVICE_LEVEL", "0.9")
    monkeypatch.setenv("HOLDING_COST", "0.4")
    monkeypatch.setenv("STOCKOUT_COST", "5")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_JSON", "true")

    settings = load_settings()

    assert settings.forecasting.model_path == str(model_dir)
    assert settings.database.analyses_db_path == str(db_path)
    assert settings.database.database_url == "postgresql+psycopg2://supplysync:supplysync@localhost:5432/supplysync"
    assert settings.app.allowed_origins == ["http://localhost:3000", "https://demo.example.com"]
    assert settings.inventory.default_lead_time_days == 14
    assert settings.inventory.default_service_level == 0.9
    assert settings.inventory.holding_cost_per_unit == 0.4
    assert settings.inventory.stockout_cost_per_unit == 5.0
    assert settings.logging.level == "DEBUG"
    assert settings.logging.json_logs is True


def test_settings_resolve_repo_root_style_backend_paths(monkeypatch):
    from config.settings import PROJECT_ROOT, load_settings

    monkeypatch.setenv("MODEL_PATH", "backend/saved_models")
    monkeypatch.setenv("ANALYSES_DB_PATH", "backend/data/analyses.sqlite")

    settings = load_settings()

    assert Path(settings.forecasting.model_path) == PROJECT_ROOT / "backend" / "saved_models"
    assert Path(settings.database.analyses_db_path) == PROJECT_ROOT / "backend" / "data" / "analyses.sqlite"


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
