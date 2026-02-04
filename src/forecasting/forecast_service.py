# src/forecasting/forecast_service.py

import numpy as np

def forecast_next_days(model, last_features, horizon=7):
    """
    Generates demand forecast for the next N days using a trained model.
    
    This function performs recursive forecasting:
    each prediction is fed back as the next day's lag feature.
    
    Parameters:
    - model: trained LightGBM model
    - last_features: DataFrame row containing latest feature values
    - horizon: number of days to forecast
    
    Returns:
    - List of predicted demand values for next N days
    """
    
    forecasts = []
    current_features = last_features.copy()
    
    for _ in range(horizon):
        # Predict next day's demand
        pred = model.predict(current_features)[0]
        forecasts.append(pred)
        
        # Update lag feature for next iteration
        # (simplified recursive update)
        current_features["lag_1"] = pred
    
    return forecasts
