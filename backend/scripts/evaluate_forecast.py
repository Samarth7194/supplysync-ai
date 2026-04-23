"""Backtest the forecasting stack against simple baselines.

For every top-N SKU we:

  1. Split the series temporally — last ``HORIZON`` days are the holdout.
  2. Classify demand pattern (regular / intermittent / highly_intermittent).
  3. Run one-step-ahead predictions over the holdout for every model below,
     feeding the real previous-day actuals back each step:
        - lightgbm    (trained model from saved_models/)
        - naive_last
        - seasonal_naive_7
        - moving_avg_7
        - croston_sba
  4. Aggregate MAE, RMSE, bias, WAPE, and MASE per SKU and per demand class.

Outputs ``backend/data/forecast_evaluation.json`` and
``forecast_evaluation.csv`` for reference in the README and dashboards.

Usage:
    cd backend
    python scripts/evaluate_forecast.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

from ingestion.load_retail_data import load_sku_demand  # noqa: E402
from features.inference_features import build_inference_features  # noqa: E402
from services.adaptive_forecasting_service import classify_sku_demand_pattern  # noqa: E402
from services.model_service import get_model_service  # noqa: E402
from evaluation import baselines  # noqa: E402
from evaluation.metrics import compute_all  # noqa: E402


HORIZON = 30               # days held out per SKU for evaluation
MIN_HISTORY_DAYS = 60      # skip SKUs with less than this


def _load_model_and_schema():
    saved_models_dir = BACKEND_DIR / "saved_models"
    model_service = get_model_service(model_dir=str(saved_models_dir))
    try:
        model = model_service.load_model("lightgbm_demand_forecast")
    except FileNotFoundError:
        return None, None
    meta = model_service.get_model_metadata("lightgbm_demand_forecast") or {}
    return model, meta.get("features")


def _lightgbm_predictions(
    model,
    feature_columns: List[str],
    series: pd.Series,
    test_window: pd.Series,
) -> np.ndarray:
    """One-step-ahead LightGBM predictions over ``test_window``.

    At every step we rebuild the feature row from the history-up-to-yesterday
    so the model is evaluated on exactly the same preprocessing path the
    live analyze endpoint uses.
    """
    out = np.empty(len(test_window), dtype=float)
    running = series.loc[: test_window.index[0]].iloc[:-1]  # everything strictly before day 0

    for i, (date, actual) in enumerate(test_window.items()):
        features = build_inference_features(running, feature_columns)
        if features is None:
            out[i] = float(running.tail(7).mean()) if len(running) else 0.0
        else:
            pred = float(model.predict(features)[0])
            out[i] = max(0.0, pred)
        running = pd.concat([running, pd.Series([actual], index=[date])])
    return out


def _evaluate_sku(
    sku: str,
    series: pd.Series,
    model,
    feature_columns,
) -> Optional[dict]:
    if len(series) < MIN_HISTORY_DAYS:
        return None

    train = series.iloc[:-HORIZON]
    test = series.iloc[-HORIZON:]
    if len(test) == 0:
        return None

    demand_class = classify_sku_demand_pattern(series)
    in_sample = train.to_numpy(dtype=float)
    actual = test.to_numpy(dtype=float)

    per_model: Dict[str, dict] = {}

    # Baselines
    for name, fn in baselines.BASELINES.items():
        preds = baselines.run_baseline_over_window(train, test, fn)
        metrics = compute_all(actual, preds, in_sample=in_sample, seasonality=1)
        per_model[name] = metrics.as_dict()

    # LightGBM
    if model is not None and feature_columns:
        try:
            lgb_preds = _lightgbm_predictions(model, feature_columns, series, test)
            metrics = compute_all(actual, lgb_preds, in_sample=in_sample, seasonality=1)
            per_model["lightgbm"] = metrics.as_dict()
        except Exception as exc:
            print(f"  [{sku}] LightGBM eval failed: {exc}")

    return {
        "sku": sku,
        "demand_class": demand_class,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "total_test_demand": float(actual.sum()),
        "metrics": per_model,
    }


def _aggregate(
    sku_results: List[dict],
    model_names: List[str],
) -> Dict[str, Dict[str, dict]]:
    """Weighted averages per model, overall and per demand class.

    WAPE is reaggregated from totals so it stays a true global WAPE rather
    than an average of per-SKU WAPEs. MAE / RMSE / bias are demand-weighted
    by ``n_test`` — a SKU with 30 test days counts the same as another SKU
    with 30 test days, irrespective of volume.
    """
    buckets: Dict[str, List[dict]] = {"all": list(sku_results)}
    for r in sku_results:
        buckets.setdefault(r["demand_class"], []).append(r)

    out: Dict[str, Dict[str, dict]] = {}
    for bucket_name, rows in buckets.items():
        out[bucket_name] = {}
        for model_name in model_names:
            relevant = [r for r in rows if model_name in r["metrics"]]
            if not relevant:
                continue
            total_n = sum(r["metrics"][model_name]["n"] for r in relevant)
            if total_n == 0:
                continue

            def _wmean(key: str) -> Optional[float]:
                num = 0.0
                den = 0
                for r in relevant:
                    m = r["metrics"][model_name]
                    v = m.get(key)
                    if v is None:
                        continue
                    num += v * m["n"]
                    den += m["n"]
                return round(num / den, 4) if den else None

            # Global WAPE: sum of |err| / sum of |actual| across all rows.
            sum_abs_err = 0.0
            sum_abs_act = 0.0
            for r in relevant:
                m = r["metrics"][model_name]
                if m.get("wape") is None:
                    continue
                sum_abs_err += m["wape"] * r["total_test_demand"]
                sum_abs_act += r["total_test_demand"]
            global_wape = round(sum_abs_err / sum_abs_act, 4) if sum_abs_act > 0 else None

            out[bucket_name][model_name] = {
                "mae": _wmean("mae"),
                "rmse": _wmean("rmse"),
                "bias": _wmean("bias"),
                "wape": global_wape,
                "mase": _wmean("mase"),
                "n_skus": len(relevant),
                "n_test_points": total_n,
            }
    return out


def _write_csv(path: Path, sku_results: List[dict], model_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sku", "demand_class", "model", "mae", "rmse", "bias", "wape", "mase", "n"])
        for r in sku_results:
            for model_name in model_names:
                if model_name not in r["metrics"]:
                    continue
                m = r["metrics"][model_name]
                writer.writerow([
                    r["sku"], r["demand_class"], model_name,
                    m.get("mae"), m.get("rmse"), m.get("bias"),
                    m.get("wape"), m.get("mase"), m.get("n"),
                ])


def _print_summary(aggregated: Dict[str, Dict[str, dict]], model_names: List[str]) -> None:
    for bucket in ["all", "regular", "intermittent", "highly_intermittent"]:
        if bucket not in aggregated or not aggregated[bucket]:
            continue
        header = f"  {bucket.upper()}"
        print()
        print(header)
        print("  " + "-" * 70)
        print(f"  {'model':<20} {'MAE':>8} {'RMSE':>8} {'bias':>8} {'WAPE':>8} {'MASE':>8} {'n':>6}")
        for model_name in model_names:
            if model_name not in aggregated[bucket]:
                continue
            m = aggregated[bucket][model_name]

            def _fmt(v, suffix=""):
                return f"{v:>8.2f}{suffix}" if isinstance(v, (int, float)) else f"{'--':>8}{suffix}"

            print(
                f"  {model_name:<20} "
                f"{_fmt(m['mae'])} {_fmt(m['rmse'])} {_fmt(m['bias'])} "
                f"{_fmt(m['wape'])} {_fmt(m['mase'])} {m['n_test_points']:>6}"
            )


def main() -> int:
    print("=" * 60)
    print("SupplySync forecast evaluation")
    print("=" * 60)

    # Honor DATA_PARQUET_PATH env var for bring-your-own-data workflows, fall
    # back to the default project location otherwise.
    parquet_env = os.environ.get("DATA_PARQUET_PATH")
    parquet_path = Path(parquet_env) if parquet_env else BACKEND_DIR.parent / "data" / "processed" / "daily_demand.parquet"
    if not parquet_path.exists():
        print(f"Processed dataset missing: {parquet_path}")
        print("Run `python scripts/bootstrap.py` first.")
        return 1

    model, feature_columns = _load_model_and_schema()
    if model is None:
        print("Trained model not found; evaluating baselines only.")

    daily = pd.read_parquet(parquet_path)
    sku_stats = daily.groupby("StockCode").agg(n=("date", "nunique"), total=("demand", "sum"))
    sku_stats = sku_stats[sku_stats["n"] >= MIN_HISTORY_DAYS]
    top_skus = sku_stats.nlargest(20, "total").index.tolist()
    print(
        f"Evaluating {len(top_skus)} SKUs (>= {MIN_HISTORY_DAYS}-day history) "
        f"on a {HORIZON}-day holdout."
    )

    sku_results: List[dict] = []
    for sku in top_skus:
        series_df = load_sku_demand(sku)
        if series_df.empty:
            continue
        series = pd.Series(
            series_df["demand"].values, index=pd.DatetimeIndex(series_df["date"])
        )
        res = _evaluate_sku(sku, series, model, feature_columns)
        if res is None:
            continue
        sku_results.append(res)
        print(f"  {sku}  class={res['demand_class']:<20} n_test={res['n_test']}")

    if not sku_results:
        print("No SKUs produced evaluable windows.")
        return 1

    model_names = list(baselines.BASELINES.keys())
    if model is not None:
        model_names.append("lightgbm")

    aggregated = _aggregate(sku_results, model_names)
    _print_summary(aggregated, model_names)

    out_dir = BACKEND_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "forecast_evaluation.json"
    csv_path = out_dir / "forecast_evaluation.csv"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "horizon_days": HORIZON,
        "n_skus_evaluated": len(sku_results),
        "models": model_names,
        "aggregates": aggregated,
        "per_sku": sku_results,
    }
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    _write_csv(csv_path, sku_results, model_names)

    print(f"\nSaved:\n  {json_path}\n  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
