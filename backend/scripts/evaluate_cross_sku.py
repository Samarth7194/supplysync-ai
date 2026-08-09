"""Cross-SKU generalization evaluation.

Holds out groups of SKUs the model never saw during training, retrains
LightGBM on the remaining SKUs, and compares it against four statistical
baselines on the held-out set. Answers the "does this generalize to
unseen products?" question honestly — even if baselines win.

Outputs:
    backend/data/cross_sku_evaluation.json

Usage:
    cd backend
    python scripts/evaluate_cross_sku.py --folds 5 --top-skus 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

from lightgbm import LGBMRegressor  # noqa: E402

from ingestion.load_retail_data import load_sku_demand  # noqa: E402
from features.lag_features import create_lag_features  # noqa: E402
from features.schema import FEATURE_COLUMNS  # noqa: E402
from features.time_features import create_time_features  # noqa: E402
from services.adaptive_forecasting_service import classify_sku_demand_pattern  # noqa: E402
from evaluation import baselines  # noqa: E402
from evaluation.metrics import compute_all  # noqa: E402


HORIZON = 30
MIN_HISTORY_DAYS = 60


def _prepare_sku_features(sku_df: pd.DataFrame) -> pd.DataFrame:
    df = create_lag_features(sku_df, target_col="demand")
    df = create_time_features(df, date_col="date")
    return df.dropna(subset=FEATURE_COLUMNS)


def _train_on_sku_group(
    train_skus: List[str],
) -> Optional[LGBMRegressor]:
    """Train a LightGBM on the concatenated training data for a group of SKUs."""
    train_frames: List[pd.DataFrame] = []
    for sku in train_skus:
        sku_df = load_sku_demand(sku)
        if len(sku_df) < MIN_HISTORY_DAYS:
            continue
        featured = _prepare_sku_features(sku_df)
        if len(featured) < MIN_HISTORY_DAYS // 2:
            continue
        # Drop the last HORIZON days from every training SKU so no fold's
        # holdout window is accidentally seen during training.
        featured = featured.iloc[:-HORIZON] if len(featured) > HORIZON else featured
        train_frames.append(featured)

    if not train_frames:
        return None

    combined = pd.concat(train_frames, ignore_index=True)
    model = LGBMRegressor(
        objective="regression",
        metric="mae",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    model.fit(combined[FEATURE_COLUMNS], combined["demand"])
    return model


def _lightgbm_one_step(
    model: LGBMRegressor,
    series: pd.Series,
    test_window: pd.Series,
) -> np.ndarray:
    """Walk-forward one-step-ahead LightGBM predictions with a rebuilt feature row."""
    out = np.empty(len(test_window), dtype=float)
    running = series.loc[: test_window.index[0]].iloc[:-1]

    for i, (date, actual) in enumerate(test_window.items()):
        features = _features_from_history(running)
        if features is None:
            out[i] = float(running.tail(7).mean()) if len(running) else 0.0
        else:
            pred = float(model.predict(features)[0])
            out[i] = max(0.0, pred)
        running = pd.concat([running, pd.Series([actual], index=[date])])
    return out


def _features_from_history(history: pd.Series) -> Optional[pd.DataFrame]:
    """Build a single-row feature frame from a recent history window."""
    if len(history) < 15:
        return None
    recent = history.tail(30)
    df = pd.DataFrame({"date": recent.index, "demand": recent.values})
    featured = _prepare_sku_features(df)
    if featured.empty:
        return None
    return featured[FEATURE_COLUMNS].iloc[[-1]]


def _evaluate_on_holdout(
    sku: str,
    series: pd.Series,
    model: Optional[LGBMRegressor],
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
    for name, fn in baselines.BASELINES.items():
        preds = baselines.run_baseline_over_window(train, test, fn)
        per_model[name] = compute_all(actual, preds, in_sample=in_sample, seasonality=1).as_dict()

    if model is not None:
        try:
            lgb_preds = _lightgbm_one_step(model, series, test)
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


def _aggregate(per_sku_results: List[dict], model_names: List[str]) -> Dict[str, dict]:
    """Demand-weighted means across all held-out SKUs and folds."""
    if not per_sku_results:
        return {}
    out: Dict[str, dict] = {}
    for model_name in model_names:
        relevant = [r for r in per_sku_results if model_name in r["metrics"]]
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

        out[model_name] = {
            "mae": _wmean("mae"),
            "rmse": _wmean("rmse"),
            "bias": _wmean("bias"),
            "mase": _wmean("mase"),
            "n_skus": len(relevant),
            "n_test_points": total_n,
        }
    return out


def _winner_rate(per_sku_results: List[dict], model_names: List[str]) -> Dict[str, float]:
    """Fraction of SKUs on which each model had the lowest MAE."""
    wins = {m: 0 for m in model_names}
    total = 0
    for r in per_sku_results:
        scoreboard = []
        for m in model_names:
            if m in r["metrics"] and r["metrics"][m].get("mae") is not None:
                scoreboard.append((m, r["metrics"][m]["mae"]))
        if not scoreboard:
            continue
        total += 1
        winner = min(scoreboard, key=lambda x: x[1])[0]
        wins[winner] += 1
    return {m: round(wins[m] / total, 3) if total else 0.0 for m in model_names}


def _print_summary(fold_summaries: List[dict], aggregate: Dict[str, dict], winner: Dict[str, float]) -> None:
    print()
    print("=" * 72)
    print("  Cross-SKU generalization summary")
    print("=" * 72)
    print(f"  {'model':<20} {'MAE':>10} {'RMSE':>10} {'bias':>10} {'MASE':>10} {'winner%':>10}")
    print("  " + "-" * 70)

    def _fmt(v):
        return f"{v:>10.2f}" if isinstance(v, (int, float)) else f"{'--':>10}"

    for model_name, stats in aggregate.items():
        print(
            f"  {model_name:<20} "
            f"{_fmt(stats['mae'])} {_fmt(stats['rmse'])} {_fmt(stats['bias'])} "
            f"{_fmt(stats['mase'])} {winner.get(model_name, 0.0):>10.0%}"
        )
    print()
    print("  Per-fold held-out SKUs:")
    for i, fs in enumerate(fold_summaries, 1):
        print(f"    fold {i}: {fs['holdout_skus']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-SKU generalization evaluation.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--top-skus", type=int, default=20, help="Top SKUs by total demand.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the JSON report. Defaults to backend/data/cross_sku_evaluation.json.",
    )
    args = parser.parse_args()

    if args.folds < 2:
        print("--folds must be >= 2.")
        return 1

    parquet_env = os.environ.get("DATA_PARQUET_PATH")
    parquet_path = Path(parquet_env) if parquet_env else BACKEND_DIR.parent / "data" / "processed" / "daily_demand.parquet"
    if not parquet_path.exists():
        print(f"Processed dataset missing: {parquet_path}")
        print("Run `python scripts/bootstrap.py` first.")
        return 1

    print("=" * 72)
    print("SupplySync cross-SKU generalization evaluation")
    print("=" * 72)
    print(f"  folds={args.folds}  top_skus={args.top_skus}  horizon={HORIZON}")

    daily = pd.read_parquet(parquet_path)
    sku_stats = daily.groupby("StockCode").agg(n=("date", "nunique"), total=("demand", "sum"))
    sku_stats = sku_stats[sku_stats["n"] >= MIN_HISTORY_DAYS]
    top = sku_stats.nlargest(args.top_skus, "total").index.tolist()
    if len(top) < args.folds:
        print(f"Not enough SKUs ({len(top)}) for {args.folds} folds.")
        return 1

    # Deterministic fold assignment — sort by demand, then round-robin
    # (so each fold has a mix of heavy + lighter SKUs).
    folds: List[List[str]] = [[] for _ in range(args.folds)]
    for i, sku in enumerate(top):
        folds[i % args.folds].append(sku)

    all_results: List[dict] = []
    fold_summaries: List[dict] = []

    for fold_idx, holdout in enumerate(folds, 1):
        train_skus = [s for s in top if s not in holdout]
        print(f"\n[Fold {fold_idx}/{args.folds}] holdout={holdout}")
        model = _train_on_sku_group(train_skus)
        if model is None:
            print("  training produced no model — skipping fold.")
            continue

        fold_rows: List[dict] = []
        for sku in holdout:
            series_df = load_sku_demand(sku)
            if series_df.empty:
                continue
            series = pd.Series(
                series_df["demand"].values,
                index=pd.DatetimeIndex(series_df["date"]),
            )
            res = _evaluate_on_holdout(sku, series, model)
            if res is None:
                continue
            res["fold"] = fold_idx
            res["train_skus"] = train_skus
            fold_rows.append(res)
            all_results.append(res)
            print(f"  {sku}  class={res['demand_class']:<20} n_test={res['n_test']}")

        fold_summaries.append({
            "fold": fold_idx,
            "holdout_skus": holdout,
            "n_skus_evaluated": len(fold_rows),
        })

    if not all_results:
        print("No fold produced evaluable SKUs.")
        return 1

    model_names = list(baselines.BASELINES.keys()) + ["lightgbm"]
    aggregate = _aggregate(all_results, model_names)
    winner = _winner_rate(all_results, model_names)
    _print_summary(fold_summaries, aggregate, winner)

    # Per-demand-class aggregates so readers can see where LightGBM wins/loses.
    by_class: Dict[str, Dict[str, dict]] = {}
    for demand_class in {r["demand_class"] for r in all_results}:
        subset = [r for r in all_results if r["demand_class"] == demand_class]
        by_class[demand_class] = _aggregate(subset, model_names)

    output_path = Path(args.output) if args.output else BACKEND_DIR / "data" / "cross_sku_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "folds": args.folds,
        "top_skus": args.top_skus,
        "horizon_days": HORIZON,
        "model_names": model_names,
        "aggregate": aggregate,
        "aggregate_by_class": by_class,
        "winner_rate": winner,
        "fold_summaries": fold_summaries,
        "per_sku": [
            {
                "sku": r["sku"],
                "fold": r["fold"],
                "demand_class": r["demand_class"],
                "n_test": r["n_test"],
                "total_test_demand": r["total_test_demand"],
                "metrics": r["metrics"],
            }
            for r in all_results
        ],
    }
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
