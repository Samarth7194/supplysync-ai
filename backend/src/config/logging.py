"""Logging settings.

The app does not yet install structured JSON logging globally. Keeping these
settings explicit prepares that change without mixing logging concerns into
database or inventory configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.common import env_bool


_ALLOWED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    json_logs: bool


def load_logging_settings() -> LoggingSettings:
    level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if level not in _ALLOWED_LEVELS:
        allowed = ", ".join(sorted(_ALLOWED_LEVELS))
        raise ValueError(f"LOG_LEVEL must be one of {allowed}; got {level!r}.")
    return LoggingSettings(
        level=level,
        json_logs=env_bool("LOG_JSON", False),
    )

