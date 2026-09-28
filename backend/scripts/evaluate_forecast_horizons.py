"""Alias for ``evaluate_forecast.py`` (kept so existing commands and docs still work).

Both scripts run the same shared multi-step, rolling-origin backtest
(``evaluation.backtest``). ``evaluate_forecast.py`` already writes
``forecast_evaluation_horizons.json`` for every requested horizon.

Usage:
    cd backend
    python scripts/evaluate_forecast_horizons.py            # horizons 7,14
    FORECAST_EVAL_HORIZONS=7,14,30 python scripts/evaluate_forecast_horizons.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_forecast  # noqa: E402


def main() -> int:
    raw = os.environ.get("FORECAST_EVAL_HORIZONS")
    argv = ["--horizons", raw] if raw else []
    return evaluate_forecast.main(argv + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
