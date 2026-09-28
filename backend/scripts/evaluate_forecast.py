"""Multi-step, rolling-origin backtest of the production forecast stack.

Runs ``evaluation.backtest`` (the single shared implementation) over the last
``--eval-days`` days of the dataset for one or more horizons and writes:

  * backend/data/forecast_evaluation.json          (primary horizon)
  * backend/data/forecast_evaluation_horizons.json (every requested horizon)
  * matching flat CSV files

At every forecast origin only data up to that origin is visible, the whole
horizon is forecast in one shot, and the result is scored against the real
demand that followed. Reference rows (predict-zero, 7-day moving average,
seasonal naive, Croston-SBA) are always included. A WAPE of 1.0 is what
predict-zero scores.

Usage:
    cd backend
    python scripts/evaluate_forecast.py                       # horizons 7,14
    python scripts/evaluate_forecast.py --horizons 7 --max-skus 200
    python scripts/evaluate_forecast.py --model tweedie=saved_models/candidates/lightgbm_demand_forecast_tweedie.pkl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

from config.settings import load_settings  # noqa: E402
from evaluation import backtest as bt  # noqa: E402
from features.schema import feature_columns_for_version, FEATURE_SCHEMA_VERSION  # noqa: E402

DEFAULT_HORIZONS = [7, 14]
DEFAULT_MAX_SKUS = 500
DEFAULT_EVAL_DAYS = 30


def _load_artifact(pkl_path: Path) -> tuple[object, list[str], dict]:
    """Load a pickled model plus its metadata, verifying the recorded checksum."""
    metadata_path = pkl_path.with_name(pkl_path.name.replace(".pkl", "_metadata.json"))
    if not metadata_path.exists():
        # Production artifact keeps a single metadata file name.
        metadata_path = pkl_path.with_name("lightgbm_demand_forecast_metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    digest = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
    expected = metadata.get("artifact_checksum")
    if expected and expected != digest:
        raise SystemExit(f"Checksum mismatch for {pkl_path.name}: metadata says {expected[:12]}, file is {digest[:12]}")
    with pkl_path.open("rb") as fh:
        model = pickle.load(fh)
    version = metadata.get("feature_schema_version") or FEATURE_SCHEMA_VERSION
    columns = metadata.get("features") or feature_columns_for_version(version)
    return model, list(columns), {
        "version": metadata.get("version"),
        "feature_schema_version": version,
        "artifact_checksum": digest,
        "training_config": metadata.get("training_config"),
    }


def _flat_rows(payloads: dict[str, dict]) -> list[list]:
    rows = []
    for horizon, payload in payloads.items():
        for bucket, methods in payload["aggregates"].items():
            for method, m in methods.items():
                rows.append([
                    horizon, bucket, method, m["wape"], m["wape_lead_time_sum"], m["mae"], m["rmse"],
                    m["bias"], m["bias_ratio"], m["mase"], m["n_skus"], m["n_origins"], m["n_test_points"],
                ])
    return rows


def _write_csv(path: Path, payloads: dict[str, dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "horizon_days", "demand_class", "method", "wape_daily", "wape_lead_time_sum", "mae", "rmse",
            "bias", "bias_ratio", "mase", "n_skus", "n_origins", "n_test_points",
        ])
        writer.writerows(_flat_rows(payloads))


def build_forecasters(args: argparse.Namespace, artifacts: dict[str, tuple]) -> dict:
    forecasters = {}
    prod = artifacts.get("production")
    if prod is not None:
        model, columns, _ = prod
        forecasters[bt.PRODUCTION_ROUTED] = bt.production_forecaster(model, columns)
        forecasters[bt.LIGHTGBM] = bt.lightgbm_forecaster(model, columns)
    for label, (model, columns, _) in artifacts.items():
        if label == "production":
            continue
        forecasters[f"lightgbm_{label}"] = bt.lightgbm_forecaster(model, columns)
    return forecasters


def compare_to_strongest_baseline(result: bt.BacktestResult, *, n_boot: int = 300) -> dict:
    """Per class: who is the strongest baseline, and does each model beat it (with a paired 95% CI)?"""
    comparisons: dict = {}
    for bucket in ("all", *bt.DEMAND_CLASSES):
        methods = result.aggregates.get(bucket) or {}
        strongest = bt.strongest_baseline(methods)
        if strongest is None:
            continue
        entry = {
            "metric": "wape_lead_time_sum",
            "strongest_baseline": {"method": strongest[0], "value": strongest[1]},
            "methods": {},
        }
        for name, metrics in methods.items():
            if name in bt.REFERENCE_METHODS:
                continue
            value = metrics.get("wape_lead_time_sum")
            entry["methods"][name] = {
                "beats_strongest_baseline": value is not None and value < strongest[1],
                "difference_vs_strongest_baseline": bt.bootstrap_wape_ci(
                    result.per_sku, demand_class=bucket, method=name, reference=strongest[0], n_boot=n_boot
                ),
            }
        comparisons[bucket] = entry
    return comparisons


def run(args: argparse.Namespace) -> dict[str, dict]:
    parquet_env = os.environ.get("DATA_PARQUET_PATH")
    parquet_path = Path(parquet_env) if parquet_env else BACKEND_DIR.parent / "data" / "processed" / "daily_demand.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"Processed dataset missing: {parquet_path}\nRun `python scripts/bootstrap.py` first.")

    daily = pd.read_parquet(parquet_path)
    daily["date"] = pd.to_datetime(daily["date"])
    dataset_end = daily["date"].max()
    cutoff = dataset_end - pd.Timedelta(days=args.eval_days)

    skus = bt.select_eval_skus(daily, cutoff=cutoff, min_active_days=60, max_skus=args.max_skus)
    series = bt.prepare_sku_series(daily, skus=skus, pad_to=dataset_end)
    print(f"Backtest: {len(series)} SKUs, dataset_end={dataset_end.date()}, train_cutoff={cutoff.date()}, "
          f"eval_days={args.eval_days}, stride={args.stride}")

    artifacts: dict[str, tuple] = {}
    prod_path = BACKEND_DIR / "saved_models" / "lightgbm_demand_forecast.pkl"
    if prod_path.exists():
        artifacts["production"] = _load_artifact(prod_path)
    else:
        print("Production model artifact not found; evaluating reference methods only.")
    for spec in args.model or []:
        label, _, raw = spec.partition("=")
        path = Path(raw)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        artifacts[label] = _load_artifact(path)

    forecasters = build_forecasters(args, artifacts)
    model_info = {label: info for label, (_, _, info) in artifacts.items()}

    payloads: dict[str, dict] = {}
    for horizon in args.horizons:
        config = bt.BacktestConfig(horizon=horizon, eval_days=args.eval_days, stride=args.stride)
        print(f"\n=== horizon {horizon} ===")
        result = bt.run_backtest(
            series, forecasters, config, dataset_end=dataset_end,
            progress=lambda done, total: print(f"  {done}/{total} SKUs", end="\r") if done % 50 == 0 or done == total else None,
        )
        print()
        payloads[str(horizon)] = result.to_payload(extra={
            "sku_selection": {
                "min_active_days": 60,
                "max_skus": args.max_skus,
                "ranked_on": "total demand on or before train_cutoff",
                "series_padded_to_dataset_end": True,
            },
            "model_artifacts": model_info,
            "comparisons": compare_to_strongest_baseline(result),
        })
        _print_summary(result)
    return payloads


def _print_summary(result: bt.BacktestResult) -> None:
    print(f"  origins scored: {result.n_origins} (skipped {result.n_skipped_origins}); "
          f"routed methods: {result.routed_method_counts}")
    for bucket in ("all", *bt.DEMAND_CLASSES):
        methods = result.aggregates.get(bucket)
        if not methods:
            continue
        print(f"  [{bucket}]  {'method':<26}{'WAPE':>8}{'WAPE(sum)':>11}{'bias%':>8}{'MASE':>8}{'origins':>9}")
        for name, m in sorted(methods.items(), key=lambda kv: (kv[1]['wape_lead_time_sum'] is None, kv[1]['wape_lead_time_sum'])):
            bias = f"{m['bias_ratio'] * 100:+.0f}" if m["bias_ratio"] is not None else "--"
            print(f"       {name:<26}{_f(m['wape']):>8}{_f(m['wape_lead_time_sum']):>11}{bias:>8}{_f(m['mase']):>8}{m['n_origins']:>9}")


def _f(value) -> str:
    return "--" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizons", type=lambda s: [int(v) for v in s.split(",")], default=DEFAULT_HORIZONS)
    parser.add_argument("--max-skus", type=int, default=DEFAULT_MAX_SKUS)
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--model", action="append", metavar="LABEL=PKL", help="Extra LightGBM artifact to score as lightgbm_<LABEL>.")
    parser.add_argument("--no-write", action="store_true", help="Print results without touching backend/data.")
    args = parser.parse_args(argv)

    payloads = run(args)
    if args.no_write:
        return 0

    primary = load_settings().inventory.default_lead_time_days
    primary_key = str(primary) if str(primary) in payloads else next(iter(payloads))
    out_dir = BACKEND_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forecast_evaluation.json").write_text(json.dumps(payloads[primary_key], indent=2))
    (out_dir / "forecast_evaluation_horizons.json").write_text(json.dumps({
        "schema_version": bt.SCHEMA_VERSION,
        "evaluation_mode": bt.EVALUATION_MODE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons_requested": args.horizons,
        "horizons": payloads,
    }, indent=2))
    _write_csv(out_dir / "forecast_evaluation.csv", {primary_key: payloads[primary_key]})
    _write_csv(out_dir / "forecast_evaluation_horizons.csv", payloads)
    print(f"\nSaved backend/data/forecast_evaluation*.json/.csv (primary horizon {primary_key})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
