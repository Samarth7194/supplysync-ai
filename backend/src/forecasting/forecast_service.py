"""Recursive demand forecasting using trained ML models."""

import numpy as np


def forecast_next_days(model, last_features, horizon=7):
    """
    Generates demand forecast for the next N days using a trained model.

    Uses recursive forecasting: each prediction feeds back as the next day's
    lag feature, cascading through all lag columns.
    """
    forecasts = []
    current_features = last_features.copy()

    for _ in range(horizon):
        pred = max(0, float(model.predict(current_features)[0]))
        forecasts.append(pred)

        # Cascade lag features: lag_7 <- lag_6 <- ... <- lag_1 <- pred
        for lag in range(7, 1, -1):
            col = f"lag_{lag}"
            prev_col = f"lag_{lag - 1}"
            if col in current_features.columns and prev_col in current_features.columns:
                current_features[col] = current_features[prev_col].values
        if "lag_1" in current_features.columns:
            current_features["lag_1"] = pred

        # Update rolling statistics from recent predictions
        if "rolling_mean_7" in current_features.columns:
            recent = forecasts[-7:] if len(forecasts) >= 7 else forecasts
            current_features["rolling_mean_7"] = np.mean(recent)
        if "rolling_std_7" in current_features.columns:
            recent = forecasts[-7:] if len(forecasts) >= 7 else forecasts
            current_features["rolling_std_7"] = np.std(recent) if len(recent) > 1 else 0.0
        if "rolling_mean_14" in current_features.columns:
            recent_14 = forecasts[-14:] if len(forecasts) >= 14 else forecasts
            current_features["rolling_mean_14"] = np.mean(recent_14)

    return forecasts
