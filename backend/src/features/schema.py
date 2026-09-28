"""Authoritative feature schema for the LightGBM demand model."""

from __future__ import annotations

import hashlib
import json


FEATURE_SCHEMA_VERSION = "demand_lag_calendar_v1"

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "lag_6",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "day_of_week",
    "month",
    "is_weekend",
    "day_of_month",
    "week_of_year",
]


# Every schema an artifact may declare. Old artifacts keep loading because each
# one is validated against the schema *it* was trained with, not just the newest.
FEATURE_SCHEMAS: dict[str, list[str]] = {
    "demand_lag_calendar_v1": FEATURE_COLUMNS,
}


def feature_columns_for_version(version: str) -> list[str]:
    """Ordered feature columns for a known schema version."""
    try:
        return list(FEATURE_SCHEMAS[version])
    except KeyError as exc:
        raise KeyError(f"Unknown feature schema version: {version!r}") from exc


def feature_schema_checksum(feature_columns: list[str] | tuple[str, ...] = tuple(FEATURE_COLUMNS)) -> str:
    """Return a deterministic hash for the ordered feature schema."""
    payload = {
        "version": FEATURE_SCHEMA_VERSION,
        "columns": list(feature_columns),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
