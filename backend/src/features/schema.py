"""Authoritative, versioned feature schemas for the LightGBM demand model.

Each schema is immutable once released and stays registered forever: an
artifact declares the version it was trained with and is validated against
*that* version (``validate_feature_schema``), so an older artifact — including
whichever one is currently active in production — keeps loading correctly
after a newer schema is introduced. ``FEATURE_SCHEMA_VERSION`` / ``FEATURE_COLUMNS``
name the latest schema, which new training runs use by default.

  * ``demand_lag_calendar_v1``     - 7 lags, 3 rolling stats, 5 calendar features.
  * ``demand_lag_calendar_sku_v2`` - v1 plus trailing 28-day SKU demand-profile
    features (zero share, mean, coefficient of variation) — added in the Step 3
    model-quality pass to give the model SKU-level context it previously had no
    way to see.
"""

from __future__ import annotations

import hashlib
import json

FEATURE_SCHEMA_V1 = "demand_lag_calendar_v1"
FEATURE_SCHEMA_V2 = "demand_lag_calendar_sku_v2"

FEATURE_COLUMNS_V1 = [
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

SKU_FEATURE_COLUMNS = ["zero_share_28", "mean_28", "cv_28"]
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS_V1 + SKU_FEATURE_COLUMNS

FEATURE_SCHEMAS: dict[str, list[str]] = {
    FEATURE_SCHEMA_V1: FEATURE_COLUMNS_V1,
    FEATURE_SCHEMA_V2: FEATURE_COLUMNS_V2,
}

# The newest schema. New training runs use this; artifacts already trained on
# an older schema keep validating against the version *they* declare (see
# ``validate_feature_schema`` below), not against this constant.
FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_V2
FEATURE_COLUMNS = FEATURE_COLUMNS_V2


def feature_columns_for_version(version: str) -> list[str]:
    """Ordered feature columns for a known schema version."""
    try:
        return list(FEATURE_SCHEMAS[version])
    except KeyError as exc:
        raise KeyError(f"Unknown feature schema version: {version!r}") from exc


def schema_version_for_columns(columns: list[str] | tuple[str, ...]) -> str | None:
    for version, expected in FEATURE_SCHEMAS.items():
        if list(columns) == expected:
            return version
    return None


def feature_schema_checksum(
    feature_columns: list[str] | tuple[str, ...] = tuple(FEATURE_COLUMNS),
    version: str | None = None,
) -> str:
    """Deterministic hash for an ordered feature schema.

    The version is part of the hash. When not given, it's inferred from the
    columns (falling back to the latest schema), so checksums recorded for
    older artifacts keep validating unchanged.
    """
    resolved = version or schema_version_for_columns(feature_columns) or FEATURE_SCHEMA_VERSION
    payload = {"version": resolved, "columns": list(feature_columns)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FeatureSchemaError(ValueError):
    """An artifact's declared feature schema is unknown or internally inconsistent."""


def validate_feature_schema(
    version: str | None,
    columns: list[str] | tuple[str, ...] | None = None,
    checksum: str | None = None,
) -> str:
    """Validate an artifact's declared schema against the registry; return its version.

    Any of the three inputs may be omitted (older metadata sometimes lacks
    them); whichever are present must agree with each other and with the
    registered schema they resolve to. Metadata with no version and no
    columns is assumed to predate schema versioning and is treated as v1 —
    exactly what every artifact trained before this change declares once
    re-saved, and consistent with what validation already required of them.
    """
    resolved = version
    if resolved is None and columns is not None:
        resolved = schema_version_for_columns(columns)
    if resolved is None:
        resolved = FEATURE_SCHEMA_V1
    if resolved not in FEATURE_SCHEMAS:
        raise FeatureSchemaError(f"Unknown feature schema version {resolved!r}; known versions: {sorted(FEATURE_SCHEMAS)}")
    expected = FEATURE_SCHEMAS[resolved]
    if columns is not None and list(columns) != expected:
        raise FeatureSchemaError(f"Artifact feature columns do not match registered schema {resolved!r}")
    if checksum is not None and checksum != feature_schema_checksum(expected, resolved):
        raise FeatureSchemaError(f"Artifact feature schema checksum does not match registered schema {resolved!r}")
    return resolved
