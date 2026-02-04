# src/api/routes/kpis.py

from fastapi import APIRouter

router = APIRouter()

@router.get("/kpis")
def kpis():
    """
    Returns system-level KPIs.
    """
    
    # For now, return precomputed or cached KPIs
    return {
        "total_cost": 125000,
        "fill_rate": 0.96,
        "cost_savings_pct": 17.8
    }
