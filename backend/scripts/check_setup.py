"""Report on setup readiness for the SupplySync backend.

Run after cloning to see which artifacts are present and which need to be
generated. Exits 0 if everything the API needs is in place, 1 otherwise.

Usage:
    cd backend
    python scripts/check_setup.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


@dataclass
class Check:
    name: str
    path: Path
    required_for: str
    how_to_fix: str


def _status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.is_dir() and not any(path.iterdir()):
        return "EMPTY"
    return "OK"


def _print_row(label: str, status: str, path: Path) -> None:
    symbol = {"OK": "[OK]    ", "MISSING": "[MISSING]", "EMPTY": "[EMPTY] "}[status]
    rel = path.relative_to(REPO_ROOT) if path.is_absolute() and REPO_ROOT in path.parents else path
    print(f"  {symbol}  {label:<32} {rel}")


def _check_python_deps() -> bool:
    required = ["pandas", "numpy", "lightgbm", "scipy", "sklearn", "fastapi", "pydantic"]
    missing: List[str] = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"  [MISSING]  Python deps: {', '.join(missing)}")
        print(f"             Fix: pip install -r {BACKEND_DIR.name}/requirements.txt")
        return False
    print(f"  [OK]       Python deps      ({', '.join(required)})")
    return True


def _check_model_integrity() -> bool:
    sys.path.insert(0, str(BACKEND_DIR / "src"))
    from services.model_service import ModelService  # noqa: PLC0415

    service = ModelService(model_dir=str(BACKEND_DIR / "saved_models"))
    status = service.artifact_status("lightgbm_demand_forecast")
    if status["valid"]:
        print(
            "  [OK]       Model integrity "
            f"(version={status.get('version')}, feature_schema={status.get('feature_schema_version')})"
        )
        return True
    print(f"  [MISSING]  Model integrity check failed: {status.get('error')}")
    print("             Fix: cd backend && python scripts/train_model.py")
    return False


def main() -> int:
    print("SupplySync setup check")
    print("=" * 60)

    checks = [
        Check(
            name="Raw dataset CSV",
            path=REPO_ROOT / "data" / "raw" / "online_retail_II.csv",
            required_for="training + simulation",
            how_to_fix=(
                "Download from https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
                "             and place at data/raw/online_retail_II.csv"
            ),
        ),
        Check(
            name="Processed daily demand",
            path=REPO_ROOT / "data" / "processed" / "daily_demand.parquet",
            required_for="/api/skus*, /api/analyze historical path",
            how_to_fix="cd backend && python scripts/train_model.py",
        ),
        Check(
            name="Trained LightGBM model",
            path=BACKEND_DIR / "saved_models" / "lightgbm_demand_forecast.pkl",
            required_for="regular-demand ML forecasts",
            how_to_fix="cd backend && python scripts/train_model.py",
        ),
        Check(
            name="Model metadata",
            path=BACKEND_DIR / "saved_models" / "lightgbm_demand_forecast_metadata.json",
            required_for="feature schema at inference",
            how_to_fix="cd backend && python scripts/train_model.py",
        ),
        Check(
            name="Cached KPIs",
            path=BACKEND_DIR / "data" / "cached_kpis.json",
            required_for="/api/kpis",
            how_to_fix="cd backend && python scripts/compute_kpis.py",
        ),
    ]

    all_ok = True
    deps_ok = _check_python_deps()
    all_ok = all_ok and deps_ok

    print()
    print("Artifacts:")
    for chk in checks:
        status = _status(chk.path)
        _print_row(chk.name, status, chk.path)
        if status != "OK":
            print(f"             Needed for: {chk.required_for}")
            print(f"             Fix: {chk.how_to_fix}")
            all_ok = False

    all_ok = all_ok and _check_model_integrity()

    print()
    if all_ok:
        print("All artifacts are present. You can run:")
        print("    uvicorn main:app --reload --port 8000")
        return 0

    print("Some artifacts are missing. Run `python scripts/bootstrap.py` to")
    print("generate everything that can be generated automatically, then start")
    print("the API with `uvicorn main:app --reload --port 8000`.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
