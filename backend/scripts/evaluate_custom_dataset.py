"""Evaluate baselines and the currently-trained LightGBM on a custom CSV.

Takes a retail-transactions CSV (with an optional column mapping for
non-UCI schemas), aggregates it to daily demand, and runs the same
temporal-split + baseline comparison as ``evaluate_forecast.py`` but on
the user's data.

Honesty guard: if the LightGBM artifact is present, it's applied
zero-shot to the custom data. The report prints a clear warning that for
a fair comparison you should retrain via
``DATA_CSV_PATH=... python scripts/train_model.py`` first.

Usage:
    cd backend
    python scripts/evaluate_custom_dataset.py --csv my.csv [--column-mapping cols.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

from ingestion.load_retail_data import (  # noqa: E402
    load_and_clean_retail_data,
    aggregate_daily_demand,
)
from features.inference_features import build_inference_features  # noqa: E402
from services.adaptive_forecasting_service import classify_sku_demand_pattern  # noqa: E402
from services.model_service import get_model_service  # noqa: E402
from evaluation import baselines  # noqa: E402
from evaluation.metrics import compute_all  # noqa: E402


HORIZON = 30
MIN_HISTORY_DAYS = 60


def _load_model_and_schema():
    service = get_model_service(model_dir=str(BACKEND_DIR / "saved_models"))
    try:
        model = service.load_model("lightgbm_demand_forecast")
    except FileNotFoundError:
        return None, None, None
    meta = service.get_model_metadata("lightgbm_demand_forecast") or {}
    return model, meta.get("features"), meta.get("dataset")


def _series_from_daily(daily: pd.DataFrame, sku: str) -> Optional[pd.Series]:
    sku_data = daily[daily["StockCode"] == sku].copy()
    if sku_data.empty:
        return None
    sku_data = sku_data.sort_values("date")
    full_range = pd.date_range(sku_data["date"].min(), sku_data["date"].max(), freq="D")
    series = (
        sku_data.set_index("date")["demand"]
        .reindex(full_range, fill_value=0)
    )
    series.index.name = "date"
    return series


def _lightgbm_one_step(model, feature_columns, series: pd.Series, test_window: pd.Series) -> np.ndarray:
    out = np.empty(len(test_window), dtype=float)
    running = series.loc[: test_window.index[0]].iloc[:-1]
    for i, (date, actual) in enumerate(test_window.items()):
        features = build_inference_features(running, feature_columns)
        if features is None:
            out[i] = float(running.tail(7).mean()) if len(running) else 0.0
        else:
            pred = float(model.predict(features)[0])
            out[i] = max(0.0, pred)
        running = pd.concat([running, pd.Series([actual], index=[date])])
    return out


def _evaluate_sku(sku: str, series: pd.Series, model, feature_columns) -> Optional[dict]:
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
    for name, fn in baselines.BASELINES.items():
        preds = baselines.run_baseline_over_window(train, test, fn)
        per_model[name] = compute_all(actual, preds, in_sample=in_sample, seasonality=1).as_dict()

    if model is not None and feature_columns:
        try:
            lgb_preds = _lightgbm_one_step(model, feature_columns, series, test)
            per_model["lightgbm"] = compute_all(
                actual, lgb_preds, in_sample=in_sample, seasonality=1
            ).as_dict()
        except Exception as exc:
            print(f"  [{sku}] LightGBM eval failed: {exc}")

    return {
        "sku": sku,
        "demand_class": demand_class,
        "n_test": int(len(test)),
        "total_test_demand": float(actual.sum()),
        "metrics": per_model,
    }


def _aggregate(results: List[dict], model_names: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for model_name in model_names:
        relevant = [r for r in results if model_name in r["metrics"]]
        if not relevant:
            continue
        total_n = sum(r["metrics"][model_name]["n"] for r in relevant)
        if total_n == 0:
            continue

        def _wmean(key: str):
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

        out[model_name] = {
            "mae": _wmean("mae"),
            "rmse": _wmean("rmse"),
            "bias": _wmean("bias"),
            "wape": _wmean("wape"),
            "mase": _wmean("mase"),
            "n_skus": len(relevant),
            "n_test_points": total_n,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate baselines + LightGBM on a custom CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the retail CSV.")
    parser.add_argument(
        "--column-mapping",
        type=str,
        default=None,
        help="Path to a {source_col: canonical_col} JSON mapping.",
    )
    parser.add_argument("--top-skus", type=int, default=20, help="Top SKUs by total demand.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the JSON report. Defaults to backend/data/custom_dataset_evaluation.json.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    mapping = None
    if args.column_mapping:
        with open(args.column_mapping) as fh:
            mapping = json.load(fh)

    print("=" * 72)
    print("SupplySync custom-dataset evaluation")
    print("=" * 72)
    print(f"  csv={csv_path}")
    if mapping:
        print(f"  column_mapping={mapping}")

    print("\n[1/3] Loading + cleaning...")
    raw = load_and_clean_retail_data(csv_path=csv_path, column_mapping=mapping)
    print(f"  cleaned rows: {len(raw):,}")

    print("[2/3] Aggregating to daily demand (in-memory only, not written to disk)...")
    daily = raw.copy()
    daily["date"] = pd.to_datetime(daily["InvoiceDate"]).dt.normalize()
    daily = daily.groupby(["StockCode", "date"])["Quantity"].sum().reset_index()
    daily.columns = ["StockCode", "date", "demand"]

    sku_stats = daily.groupby("StockCode").agg(n=("date", "nunique"), total=("demand", "sum"))
    sku_stats = sku_stats[sku_stats["n"] >= MIN_HISTORY_DAYS]
    if sku_stats.empty:
        print(f"  No SKUs have >= {MIN_HISTORY_DAYS} days of history. Aborting.")
        return 1
    top = sku_stats.nlargest(args.top_skus, "total").index.tolist()
    print(f"  Evaluating {len(top)} SKUs with >= {MIN_HISTORY_DAYS}-day history.")

    model, feature_columns, trained_dataset = _load_model_and_schema()
    zero_shot_warning = None
    if model is not None:
        zero_shot_warning = (
            f"This model was trained on '{trained_dataset or 'unknown dataset'}'. "
            f"Applying it to '{csv_path.name}' without retraining is a zero-shot test. "
            "For a fair comparison retrain via: "
            f"DATA_CSV_PATH={csv_path} python scripts/train_model.py"
        )
        print(f"\n[zero-shot warning] {zero_shot_warning}")
    else:
        print("\nNo LightGBM artifact found; evaluating baselines only.")

    print("\n[3/3] Evaluating...")
    results: List[dict] = []
    for sku in top:
        series = _series_from_daily(daily, sku)
        if series is None:
            continue
        res = _evaluate_sku(sku, series, model, feature_columns)
        if res is None:
            continue
        results.append(res)
        print(f"  {sku}  class={res['demand_class']:<20} n_test={res['n_test']}")

    if not results:
        print("No SKUs produced evaluable windows.")
        return 1

    model_names = list(baselines.BASELINES.keys())
    if model is not None:
        model_names.append("lightgbm")

    aggregate = _aggregate(results, model_names)

    output_path = Path(args.output) if args.output else BACKEND_DIR / "data" / "custom_dataset_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "column_mapping": mapping,
        "horizon_days": HORIZON,
        "n_skus_evaluated": len(results),
        "model_names": model_names,
        "zero_shot_warning": zero_shot_warning,
        "aggregate": aggregate,
        "per_sku": results,
    }
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    print()
    print("=" * 72)
    print("  Summary (demand-weighted across all evaluated SKUs)")
    print("=" * 72)
    print(f"  {'model':<20} {'MAE':>10} {'RMSE':>10} {'bias':>10} {'WAPE':>10} {'MASE':>10}")
    print("  " + "-" * 70)
    for model_name, stats in aggregate.items():
        def _fmt(v):
            return f"{v:>10.2f}" if isinstance(v, (int, float)) else f"{'--':>10}"
        print(
            f"  {model_name:<20} "
            f"{_fmt(stats['mae'])} {_fmt(stats['rmse'])} {_fmt(stats['bias'])} "
            f"{_fmt(stats['wape'])} {_fmt(stats['mase'])}"
        )

    print(f"\nSaved: {output_path}")
    if zero_shot_warning:
        print(f"\n[reminder] {zero_shot_warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
