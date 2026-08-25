"""Run controlled candidate retraining for one recommended retraining run.

This script trains and evaluates a candidate model only. It never promotes a
model or changes production inference.

Usage:
    cd backend
    python scripts/run_candidate_retraining.py --retraining-run-id 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

from config.settings import load_settings  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from services.candidate_training_service import CandidateTrainingError, CandidateTrainingService  # noqa: E402


def _fmt_metric(value) -> str:
    return "--" if value is None else f"{float(value):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate a controlled candidate model.")
    parser.add_argument("--retraining-run-id", type=int, required=True)
    parser.add_argument("--csv", type=str, default=None, help="Optional training CSV override.")
    parser.add_argument("--column-mapping", type=str, default=None, help="Optional JSON column mapping.")
    parser.add_argument("--parquet", type=str, default=None, help="Optional processed parquet path.")
    parser.add_argument("--horizon-days", type=int, default=None, help="Evaluation horizon. Defaults to the configured production lead time.")
    args = parser.parse_args()

    mapping = None
    if args.column_mapping:
        with open(args.column_mapping) as fh:
            mapping = json.load(fh)

    with SessionLocal() as session:
        service = CandidateTrainingService(session=session, settings=load_settings())
        try:
            result = service.train_candidate(
                retraining_run_id=args.retraining_run_id,
                csv_path=args.csv,
                column_mapping=mapping,
                parquet_path=args.parquet,
                horizon_days=args.horizon_days,
            )
            session.commit()
        except CandidateTrainingError as exc:
            session.commit()
            print(f"Candidate retraining failed safely: {exc}")
            print("Production model was not changed.")
            return 1

    evaluation = result.evaluation
    print(f"Retraining run: {result.retraining_run.id}")
    print(f"Baseline model: {result.retraining_run.baseline_model_artifact.version}")
    print(f"Candidate model: {result.candidate_artifact.version}")
    print()
    print(f"Candidate WAPE: {_fmt_metric(evaluation.candidate_metrics.get('wape'))}")
    print(f"Current WAPE: {_fmt_metric(evaluation.active_metrics.get('wape'))}")
    relative = evaluation.relative_wape_improvement
    print(f"Relative improvement: {'--' if relative is None else f'{relative:.2%}'}")
    print()
    for label, key in (
        ("Croston WAPE", "croston_sba"),
        ("Moving Average WAPE", "moving_avg_7"),
        ("Seasonal Naive WAPE", "seasonal_naive_7"),
    ):
        print(f"{label}: {_fmt_metric((evaluation.benchmark_metrics.get(key) or {}).get('wape'))}")
    print()
    print(f"Test points: {evaluation.test_points}")
    print(f"Promotion eligible: {'YES' if evaluation.promotion_eligible else 'NO'}")
    print(f"Eligibility reason: {evaluation.eligibility_reason}")
    print()
    print("Candidate remains lifecycle status = candidate.")
    print("Production model was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())