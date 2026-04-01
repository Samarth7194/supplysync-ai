# src/api/routes/forecast.py

from fastapi import APIRouter, HTTPException
from api.schemas import ForecastRequest, ForecastResponse
from services.model_service import get_model_service
from services.intelligent_inventory_service import IntelligentInventoryService
from services.data_service import DataService
import pandas as pd

router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """Returns demand forecast for a given SKU using adaptive forecasting."""
    try:
        model_service = get_model_service()

        try:
            model = model_service.load_model("lightgbm_demand_forecast")
        except FileNotFoundError:
            model = None

        # Load real demand history
        try:
            data_service = DataService.get_instance()
            demand_series = data_service.get_demand_history(request.sku)
            if len(demand_series) < 7:
                raise HTTPException(status_code=404, detail=f"Insufficient data for SKU {request.sku}")
            demand_history = demand_series.tail(60)
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Data not available. Run train_model.py first.")

        last_features = model_service.get_cached_features(request.sku)
        intelligent_service = IntelligentInventoryService(model=model)

        decision = intelligent_service.get_intelligent_reorder_decision(
            sku=request.sku,
            current_stock=0,
            demand_history=demand_history,
            lead_time_days=request.horizon,
            last_features=last_features,
        )

        # Extract per-day forecast from lead_time_demand
        daily_avg = decision["lead_time_demand"] / request.horizon if request.horizon > 0 else 0
        forecast_values = [round(daily_avg, 2)] * request.horizon

        return {"sku": request.sku, "forecast": forecast_values}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")
