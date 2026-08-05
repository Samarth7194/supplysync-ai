"""Composed backend settings.

Each domain owns its own settings module and validation. This file remains the
single public composition point used by the FastAPI runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from auth.session import AuthConfig
from config.app import AppSettings, load_app_settings
from config.common import BACKEND_DIR, PROJECT_ROOT
from config.database import DatabaseSettings, load_database_settings
from config.forecasting import ForecastingSettings, load_forecasting_settings
from config.inventory import InventorySettings, load_inventory_settings
from config.logging import LoggingSettings, load_logging_settings
from config.auth import load_auth_settings


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    auth: AuthConfig
    database: DatabaseSettings
    forecasting: ForecastingSettings
    inventory: InventorySettings
    logging: LoggingSettings


def load_settings() -> Settings:
    """Load and validate every runtime settings domain."""
    return Settings(
        app=load_app_settings(),
        auth=load_auth_settings(),
        database=load_database_settings(),
        forecasting=load_forecasting_settings(),
        inventory=load_inventory_settings(),
        logging=load_logging_settings(),
    )
