# src/api/routes/forecast.py

from fastapi import APIRouter
from src.api.schemas import ForecastRequest, ForecastResponse
from src.forecasting.forecast_service import forecast_next_days

router = APIRouter()

@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """
    Returns demand forecast for a given SKU.
    """
    
    # TODO (temporary):
    # Load model + last features (hardcoded or cached initially)
    # In Phase 9, we keep this simple
    
    forecast_values = forecast_next_days(
        model=model,
        last_features=last_features,
        horizon=request.horizon
    )
    
    return {
        "sku": request.sku,
        "forecast": forecast_values
    }
