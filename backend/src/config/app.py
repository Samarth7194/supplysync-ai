"""Application-level HTTP settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.common import split_csv


@dataclass(frozen=True)
class AppSettings:
    allowed_origins: list[str]


def load_app_settings() -> AppSettings:
    origins = split_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:3000"))
    if not origins:
        raise ValueError("ALLOWED_ORIGINS must contain at least one origin.")
    return AppSettings(allowed_origins=origins)

