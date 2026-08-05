"""Forecasting and model-artifact settings."""

from __future__ import annotations

from dataclasses import dataclass

from config.common import BACKEND_DIR, env_path


@dataclass(frozen=True)
class ForecastingSettings:
    model_path: str


def load_forecasting_settings() -> ForecastingSettings:
    return ForecastingSettings(
        model_path=env_path("MODEL_PATH", BACKEND_DIR / "saved_models"),
    )

