"""Persistence settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.common import BACKEND_DIR, env_path


@dataclass(frozen=True)
class DatabaseSettings:
    analyses_db_path: str
    database_url: str


def load_database_settings() -> DatabaseSettings:
    default_sqlite_url = f"sqlite:///{(BACKEND_DIR / 'data' / 'supplysync.db').as_posix()}"
    return DatabaseSettings(
        analyses_db_path=env_path(
            "ANALYSES_DB_PATH",
            BACKEND_DIR / "data" / "analyses.sqlite",
        ),
        database_url=os.getenv("DATABASE_URL", default_sqlite_url).strip(),
    )
