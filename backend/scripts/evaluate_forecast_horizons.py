"""Generate horizon-compatible offline forecast evidence.

Outputs:
  * backend/data/forecast_evaluation_horizons.json
  * backend/data/forecast_evaluation_horizons.csv

This complements the legacy 30-day ``forecast_evaluation.json`` artifact
without changing its schema.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

import evaluate_forecast as evaluator  # noqa: E402
from ingestion.load_retail_data import load_sku_demand  # noqa: E402


DEFAULT_HORIZONS = [3, 7, 14, 30]


def _evaluate_horizon(horizon: int, parquet_path: Path) -> dict:
    model, feature_columns = evaluator._load_model_and_schema()
    daily = pd.read_parquet(parquet_path)
    sku_stats = daily.groupby("StockCode").agg(n=("date", "nunique"), total=("demand", "sum"))
    sku_stats = sku_stats[sku_stats["n"] >= max(evaluator.MIN_HISTORY_DAYS, horizon + 14)]
    top_skus = sku_stats.nlargest(20, "total").index.tolist()

    sku_results = []
    for sku in top_skus:
        series_df = load_sku_demand(sku)
        if series_df.empty:
            continue
        series = pd.Series(series_df["demand"].values, index=pd.DatetimeIndex(series_df["date"]))
        result = evaluator._evaluate_sku(sku, series, model, feature_columns, horizon=horizon)
        if result is not None:
            sku_results.append(result)

    model_names = list(evaluator.baselines.BASELINES.keys())
    if model is not None:
        model_names.append("lightgbm")

    return {
        "generated_at": datetime.now().isoformat(),
        "horizon_days": horizon,
        "n_skus_evaluated": len(sku_results),
        "models": model_names,
        "aggregates": evaluator._aggregate(sku_results, model_names) if sku_results else {},
        "per_sku": sku_results,
    }


def _write_csv(path: Path, horizons: dict[str, dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["horizon_days", "scope", "model", "mae", "rmse", "bias", "wape", "mase", "n_skus", "n_test_points"])
        for horizon, payload in horizons.items():
            for scope, by_model in (payload.get("aggregates") or {}).items():
                for model, metrics in by_model.items():
                    writer.writerow(
                        [
                            horizon,
                            scope,
                            model,
                            metrics.get("mae"),
                            metrics.get("rmse"),
                            metrics.get("bias"),
                            metrics.get("wape"),
                            metrics.get("mase"),
                            metrics.get("n_skus"),
                            metrics.get("n_test_points"),
                        ]
                    )


def main() -> int:
    raw = os.environ.get("FORECAST_EVAL_HORIZONS")
    horizons = [int(v.strip()) for v in raw.split(",")] if raw else DEFAULT_HORIZONS
    parquet_env = os.environ.get("DATA_PARQUET_PATH")
    parquet_path = Path(parquet_env) if parquet_env else BACKEND_DIR.parent / "data" / "processed" / "daily_demand.parquet"
    if not parquet_path.exists():
        print(f"Processed dataset missing: {parquet_path}")
        return 1

    payload = {
        "generated_at": datetime.now().isoformat(),
        "horizons_requested": horizons,
        "horizons": {},
    }
    for horizon in horizons:
        print(f"Evaluating horizon={horizon} days")
        payload["horizons"][str(horizon)] = _evaluate_horizon(horizon, parquet_path)

    out_json = BACKEND_DIR / "data" / "forecast_evaluation_horizons.json"
    out_csv = BACKEND_DIR / "data" / "forecast_evaluation_horizons.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    _write_csv(out_csv, payload["horizons"])
    print(f"Saved:\n  {out_json}\n  {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
