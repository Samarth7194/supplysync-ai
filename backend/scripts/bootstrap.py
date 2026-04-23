"""One-shot setup for the SupplySync backend.

Generates every artifact the API needs when they are missing:

  1. data/processed/daily_demand.parquet  (via train_model.py)
  2. saved_models/lightgbm_demand_forecast.pkl + metadata.json  (via train_model.py)
  3. backend/data/cached_kpis.json  (via compute_kpis.py)

Existing artifacts are not rebuilt. Pass --force to regenerate everything.

Usage:
    cd backend
    python scripts/bootstrap.py           # generate what's missing
    python scripts/bootstrap.py --force   # regenerate everything
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
SCRIPTS_DIR = BACKEND_DIR / "scripts"

RAW_CSV = REPO_ROOT / "data" / "raw" / "online_retail_II.csv"
PROCESSED_PARQUET = REPO_ROOT / "data" / "processed" / "daily_demand.parquet"
MODEL_PKL = BACKEND_DIR / "saved_models" / "lightgbm_demand_forecast.pkl"
KPIS_JSON = BACKEND_DIR / "data" / "cached_kpis.json"


def _run(script: Path) -> None:
    print(f"\n--> Running {script.name}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=BACKEND_DIR,
    )
    if result.returncode != 0:
        print(f"\n{script.name} failed (exit {result.returncode}).")
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate artifacts even if they already exist.",
    )
    args = parser.parse_args()

    if not RAW_CSV.exists():
        print("Raw dataset is missing.")
        print(f"  Expected at: {RAW_CSV.relative_to(REPO_ROOT)}")
        print(
            "  Download from https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            "  and place the CSV at the path above, then re-run this script."
        )
        return 1

    needs_training = args.force or not MODEL_PKL.exists() or not PROCESSED_PARQUET.exists()
    needs_kpis = args.force or not KPIS_JSON.exists()

    if not needs_training and not needs_kpis:
        print("All artifacts already present. Nothing to do.")
        print("Use --force to regenerate.")
        return 0

    if needs_training:
        _run(SCRIPTS_DIR / "train_model.py")
    else:
        print(f"--> Skipping training (artifacts present at {MODEL_PKL.name})")

    if needs_kpis:
        _run(SCRIPTS_DIR / "compute_kpis.py")
    else:
        print(f"--> Skipping KPI computation (cached at {KPIS_JSON.name})")

    print("\nBootstrap complete. Start the API with:")
    print("    uvicorn main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
