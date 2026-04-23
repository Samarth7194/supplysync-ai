"""Lightweight SQLite persistence for /api/analyze snapshots.

Deliberately stdlib-only — ``sqlite3`` ships with Python — so the project
stays runnable from a clean clone without any new dependency. The file is
created on first use, lives under ``backend/data/analyses.sqlite`` by
default, and is git-ignored.

This is *local / demo* persistence: it makes the backend feel less
transient (the dashboard can show recent decisions, a reviewer can inspect
the file), not a production analytics store.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    sku                TEXT NOT NULL,
    risk               TEXT,
    action             TEXT,
    current_stock      REAL,
    recommended_order  INTEGER,
    demand_pattern     TEXT,
    forecast_method    TEXT,
    demand_source      TEXT,
    forecast_source    TEXT,
    model_type         TEXT,
    model_name         TEXT,
    artifact_available INTEGER,
    lead_time_demand   REAL,
    safety_stock       REAL,
    reorder_point      REAL,
    inventory_gap      REAL,
    p50                REAL,
    p90                REAL
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_sku        ON analyses(sku);
"""


_WRITABLE_COLUMNS = (
    "sku", "risk", "action", "current_stock", "recommended_order",
    "demand_pattern", "forecast_method", "demand_source", "forecast_source",
    "model_type", "model_name", "artifact_available",
    "lead_time_demand", "safety_stock", "reorder_point", "inventory_gap",
    "p50", "p90",
)


class AnalysisStore:
    """Thin wrapper around a SQLite file.

    Opens a fresh short-lived connection per call — simplest thing that
    works for single-process uvicorn. If this ever grows, swap in a
    connection pool or SQLModel without changing the public API.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # --- internals ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # --- public API ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def record(self, row: Mapping[str, Any]) -> int:
        """Insert a snapshot; returns the new row id.

        Unknown keys in ``row`` are ignored so callers can pass a superset.
        Booleans are coerced to 0/1 because SQLite has no native bool type.
        """
        values = []
        for col in _WRITABLE_COLUMNS:
            v = row.get(col)
            if isinstance(v, bool):
                v = 1 if v else 0
            values.append(v)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        col_sql = ", ".join(("created_at", *_WRITABLE_COLUMNS))
        qmarks = ", ".join(["?"] * (1 + len(_WRITABLE_COLUMNS)))
        sql = f"INSERT INTO analyses ({col_sql}) VALUES ({qmarks})"

        with self._connect() as conn:
            cur = conn.execute(sql, (created_at, *values))
            return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        """Most-recent analyses first. ``limit`` is clamped to [1, 200]."""
        limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,),
            )
            rows = cur.fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) AS n FROM analyses")
            return int(cur.fetchone()["n"])

    def clear(self) -> None:
        """Test helper — truncates the table. Not used by the app."""
        with self._connect() as conn:
            conn.execute("DELETE FROM analyses")


def serialize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Return rows as plain JSON-serializable dicts (safety net)."""
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        # Rehydrate booleans for cleaner JSON output.
        if "artifact_available" in d:
            d["artifact_available"] = bool(d["artifact_available"]) if d["artifact_available"] is not None else None
        out.append(d)
    return out
