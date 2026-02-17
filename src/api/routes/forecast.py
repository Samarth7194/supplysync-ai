# src/api/routes/forecast.py

from fastapi import APIRouter, HTTPException
from src.api.schemas import ForecastRequest, ForecastResponse
from src.services.model_service import get_model_service
from src.services.intelligent_inventory_service import IntelligentInventoryService

router = APIRouter()

@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """
    Returns demand forecast for a given SKU using adaptive forecasting.
    """
    
    try:
        # Get model service
        model_service = get_model_service()
        
        # Load trained model
        try:
            model = model_service.load_model("lightgbm_demand_forecast")
        except FileNotFoundError:
            raise HTTPException(
                status_code=503, 
                detail="Model not available. Please train and save the model first."
            )
        
        # Get cached features for the SKU
        last_features = model_service.get_cached_features(request.sku)
        
        if last_features is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached features found for SKU {request.sku}"
            )
        
        # Initialize intelligent service
        intelligent_service = IntelligentInventoryService(model=model)
        
        # Create dummy demand history for forecasting (in production, this would come from database)
        import pandas as pd
        import numpy as np
        
        # For demo purposes - in production, load actual historical data
        demand_history = pd.Series(np.random.poisson(10, 30))  # 30 days of dummy data
        
        # Get adaptive forecast
        forecast, method = intelligent_service.get_intelligent_reorder_decision(
            sku=request.sku,
            current_stock=0,  # Not needed for forecast only
            demand_history=demand_history,
            lead_time_days=request.horizon
        )
        
        # Extract forecast values (simplified for this endpoint)
        forecast_values = [forecast["lead_time_demand"] / request.horizon] * request.horizon
        
        # Log the forecast
        model_service.logger.info(f"Forecast generated for {request.sku} using {method}")
        
        return {
            "sku": request.sku,
            "forecast": forecast_values
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")
