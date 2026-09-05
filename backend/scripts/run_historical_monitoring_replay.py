"""Generate a historical monitoring replay snapshot.

IMPORTANT: This is historical replay evidence, not live production
monitoring. The processed dataset (data/processed/daily_demand.parquet) is a
static historical retail extract with no connected ERP/POS actual-demand
stream, so live predictions logged "today" cannot yet mature into real
forecast_evaluations — their target windows run past the dataset's end date.
This script replays the same forecast -> evaluate -> monitor lifecycle
honestly against held-out historical windows instead, so the pipeline can be
demonstrated end-to-end without fabricating live production evidence.

No model is trained. No model is promoted. No production database is
required or contacted — this reads only the local processed parquet and the
local saved-model artifact.

Usage:
    cd backend
    python scripts/run_historical_monitoring_replay.py
    python scripts/run_historical_monitoring_replay.py --horizon 14 --sku-limit 20 --num-windows 3
    python scripts/run_historical_monitoring_replay.py --json
    python scripts/run_historical_monitoring_replay.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SRC_DIR))

from config.settings import load_settings  # noqa: E402
from services.data_service import DataService  # noqa: E402
from services.historical_monitoring_replay_service import (  # noqa: E402
    DEFAULT_NUM_WINDOWS,
    DEFAULT_SKU_LIMIT,
    HistoricalMonitoringReplayError,
    HistoricalMonitoringReplayService,
)
from services.model_service import ModelService  # noqa: E402

MODEL_NAME = "lightgbm_demand_forecast"
DEFAULT_OUTPUT_PATH = BACKEND_DIR / "data" / "historical_monitoring_replay.json"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _load_model(model_dir: str) -> tuple[Any | None, list[str] | None, str | None]:
    service = ModelService(model_dir=model_dir)
    try:
        model = service.load_model(MODEL_NAME)
        meta = service.get_model_metadata(MODEL_NAME) or {}
        return model, meta.get("features"), meta.get("version")
    except Exception:  # noqa: BLE001 - replay must still run baselines/Croston without LightGBM
        return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a historical monitoring replay (not live monitoring).")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in days (default: inventory.default_lead_time_days).")
    parser.add_argument("--sku-limit", type=int, default=DEFAULT_SKU_LIMIT, help="Number of top SKUs to replay.")
    parser.add_argument("--num-windows", type=int, default=DEFAULT_NUM_WINDOWS, help="Number of sequential historical windows.")
    parser.add_argument("--model-artifact-id", type=int, default=None, help="DB model_artifacts.id to label results with, if known.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Compute the replay but do not write the output file.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Where to write the JSON result.")
    args = parser.parse_args()

    settings = load_settings()
    try:
        data_service = DataService.get_instance()
    except FileNotFoundError as exc:
        print(f"Historical replay failed: {exc}", file=sys.stderr)
        return 1

    model, feature_columns, model_version = _load_model(str(settings.forecasting.model_path))

    service = HistoricalMonitoringReplayService(
        settings=settings,
        data_service=data_service,
        model=model,
        feature_columns=feature_columns,
        model_name=MODEL_NAME,
        model_artifact_id=args.model_artifact_id,
        model_version=model_version,
        offline_evaluation_path=str(BACKEND_DIR / "data" / "forecast_evaluation.json"),
    )

    try:
        result = service.run(
            horizon_days=args.horizon,
            sku_limit=args.sku_limit,
            num_windows=args.num_windows,
        )
    except HistoricalMonitoringReplayError as exc:
        print(f"Historical replay failed: {exc}", file=sys.stderr)
        return 1

    payload = result.as_dict()

    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fh:
            json.dump(payload, fh, indent=2)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("=" * 60)
    print("SupplySync Historical Monitoring Replay")
    print("=" * 60)
    print()
    print("IMPORTANT:")
    print("This is historical replay evidence.")
    print("It is not live production monitoring.")
    print("No model was trained.")
    print("No model was promoted.")
    print()
    print(f"Model: {result.model_name}")
    print(f"Version: {_fmt(result.model_version)}")
    print(f"Horizon days: {result.horizon_days}")
    print(f"Windows replayed: {result.window_count}")
    print()
    print("Historical period")
    print(f"Start: {_fmt(result.historical_period_start)}")
    print(f"End: {_fmt(result.historical_period_end)}")
    print()
    print("Replay coverage")
    print(f"Unique SKUs evaluated across replay: {result.sku_count}")
    print(f"LightGBM artifact-scope evaluations in latest window: {result.evaluation_count}")
    print()
    print("Metrics (latest window)")
    print(f"WAPE: {_pct(result.metric_wape)}")
    print(f"Baseline WAPE: {_pct(result.baseline_wape)} (provenance: {result.baseline_provenance})")
    print(f"MAE: {_pct(result.metric_mae)}")
    print(f"RMSE: {_pct(result.metric_rmse)}")
    print(f"Bias: {_pct(result.metric_bias)}")
    print(f"MASE: {_pct(result.metric_mase)}")
    print()
    print("State")
    print(f"Status: {result.status}")
    print(f"Reason: {result.degradation_reason}")
    print(f"Message: {result.degradation_message}")
    print()
    print("Method breakdown (all windows, informational — Croston/conservative honestly represented)")
    for method, stats in result.method_breakdown.items():
        print(f"  {method:<15} skus={stats['sku_count']:<4} n={stats['evaluation_count']:<5} wape={_pct(stats['wape'])}")
    print()
    if args.dry_run:
        print("Dry run: no output file written.")
    else:
        print(f"Saved: {args.output}")
    print()
    print("PROVENANCE: historical_replay — not live production monitoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
