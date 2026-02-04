# src/api/routes/reorder.py

from fastapi import APIRouter
from src.api.schemas import ReorderRequest, ReorderResponse
from src.inventory.reorder_point import compute_reorder_decision
from src.forecasting.forecast_service import forecast_next_days

router = APIRouter()

@router.post("/reorder", response_model=ReorderResponse)
def reorder(request: ReorderRequest):
    """
    Returns reorder recommendation for a SKU.
    """
    
    forecast = forecast_next_days(model, last_features, request.lead_time_days)
    
    decision = compute_reorder_decision(
        sku=request.sku,
        current_stock=request.current_stock,
        forecast=forecast,
        sigma=sigma,
        lead_time_days=request.lead_time_days
    )
    
    return decision
