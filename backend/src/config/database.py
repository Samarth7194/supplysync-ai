"""Persistence settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.common import BACKEND_DIR


@dataclass(frozen=True)
class DatabaseSettings:
    database_url: str


def load_database_settings() -> DatabaseSettings:
    default_sqlite_url = f"sqlite:///{(BACKEND_DIR / 'data' / 'supplysync.db').as_posix()}"
    return DatabaseSettings(
        database_url=os.getenv("DATABASE_URL", default_sqlite_url).strip(),
    )
