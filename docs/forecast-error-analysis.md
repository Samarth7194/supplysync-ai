# Forecast Error Analysis

This document describes the offline forecast-error artifacts produced by:

```powershell
cd backend
python scripts/evaluate_forecast_horizons.py
python scripts/analyze_forecast_errors.py
```

## Scope

These files are **offline backtest artifacts**, not live production performance:

- `backend/data/forecast_evaluation_horizons.json`
- `backend/data/forecast_evaluation_horizons.csv`
- `backend/data/forecast_error_analysis.json`
- `backend/data/forecast_error_analysis.csv`

The evaluator uses historical SKU demand, holds out the final horizon window,
and compares runtime-capable forecasting methods against the recorded actuals.

## Methods

The benchmark includes:

- naive last value
- seasonal naive with lag 7
- moving average over 7 days
- Croston-SBA
- LightGBM, when the model artifact is available

## Metrics

- MAE: average absolute error in demand units.
- RMSE: root mean squared error, more sensitive to large misses.
- Bias: average predicted minus actual demand.
- WAPE: total absolute error divided by total actual demand.
- MASE: MAE scaled by an in-sample naive forecast.

Lower is better for MAE, RMSE, WAPE, and MASE. Bias should be near zero.

## Interpretation Rules

Do not claim model improvement unless the artifact shows it. If Croston-SBA
beats LightGBM for a horizon or demand class, report that honestly. If
LightGBM wins a segment, quantify it from the artifact rather than assuming it.

These artifacts support engineering decisions such as evidence-aware routing
and residual-informed uncertainty. They do not prove commercial cost savings.
