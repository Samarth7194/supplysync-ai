"""Shared helpers for environment-backed settings."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


def env_path(name: str, default: Path) -> str:
    raw = os.getenv(name)
    if not raw:
        return str(default)

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    # Preserve repo-root style values from .env.example, e.g.
    # backend/saved_models, regardless of the current shell directory.
    if candidate.parts and candidate.parts[0] == "backend":
        return str(PROJECT_ROOT / candidate)
    return str(candidate)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}.") from exc


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}.")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

