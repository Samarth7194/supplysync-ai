"""KPI endpoint returning real computed metrics from simulation."""

import json
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()


@router.get("/kpis")
def kpis():
    """Returns system-level KPIs computed from simulation on real data."""
    cache_path = Path(__file__).resolve().parents[3] / "data" / "cached_kpis.json"

    if not cache_path.exists():
        # Try alternate location (when running from backend/)
        alt_path = Path(__file__).resolve().parents[2] / "data" / "cached_kpis.json"
        if alt_path.exists():
            cache_path = alt_path
        else:
            return {"error": "KPIs not computed yet. Run: python scripts/compute_kpis.py"}

    with open(cache_path) as f:
        return json.load(f)
