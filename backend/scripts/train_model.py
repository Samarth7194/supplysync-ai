"""
Train a LightGBM demand forecasting model on the Online Retail II dataset.

Usage:
    cd backend
    python scripts/train_model.py
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ingestion.load_retail_data import (
    load_and_clean_retail_data,
    aggregate_daily_demand,
    get_top_skus,
    load_sku_demand,
)
from features.lag_features import create_lag_features
from features.time_features import create_time_features
from services.model_service import get_model_service, ModelService


FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
    "day_of_week", "month", "is_weekend", "day_of_month", "week_of_year",
]


def prepare_sku_features(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Create lag + time features for a single SKU's demand data."""
    df = create_lag_features(sku_df, target_col="demand")
    df = create_time_features(df, date_col="date")
    df = df.dropna(subset=FEATURE_COLUMNS)
    return df


def train():
    print("=" * 60)
    print("SupplySync AI - Model Training Pipeline")
    print("=" * 60)

    # Step 1: Load and clean data
    print("\n[1/6] Loading and cleaning raw data...")
    raw_df = load_and_clean_retail_data()
    print(f"  Cleaned data: {len(raw_df):,} rows")

    # Step 2: Aggregate to daily demand
    print("[2/6] Aggregating to daily demand...")
    daily_df = aggregate_daily_demand(raw_df)
    print(f"  Daily demand records: {len(daily_df):,}")
    print(f"  Unique SKUs: {daily_df['StockCode'].nunique():,}")

    # Step 3: Select top SKUs
    print("[3/6] Selecting top SKUs...")
    top_skus = get_top_skus(daily_df, min_days=60, top_n=20)
    print(f"  Training on {len(top_skus)} SKUs: {top_skus[:5]}...")

    # Step 4: Build feature matrix
    print("[4/6] Engineering features...")
    train_frames = []
    test_frames = []
    sku_last_features = {}

    for sku in top_skus:
        sku_df = load_sku_demand(sku)
        if len(sku_df) < 45:  # Need enough data for features + split
            continue

        featured = prepare_sku_features(sku_df)
        if len(featured) < 30:
            continue

        # Temporal split: last 30 days = test
        split_idx = len(featured) - 30
        train_frames.append(featured.iloc[:split_idx])
        test_frames.append(featured.iloc[split_idx:])

        # Cache last row of features for this SKU (for inference)
        last_row = featured[FEATURE_COLUMNS].iloc[[-1]]
        sku_last_features[sku] = last_row

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    # Step 5: Train LightGBM
    print("[5/6] Training LightGBM model...")
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["demand"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["demand"]

    model = LGBMRegressor(
        objective="regression",
        metric="mae",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)  # Clamp to non-negative

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    # MAPE only on non-zero actuals
    non_zero = y_test > 0
    mape = np.mean(np.abs((y_test[non_zero] - preds[non_zero]) / y_test[non_zero])) * 100 if non_zero.any() else 0

    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.1f}%")

    # Step 6: Save model
    print("[6/6] Saving model and caching features...")
    # Resolve saved_models dir relative to backend/
    saved_models_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    model_service = get_model_service(model_dir=saved_models_dir)

    model_service.save_model(model, "lightgbm_demand_forecast", metadata={
        "features": FEATURE_COLUMNS,
        "train_skus": top_skus,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 1),
        "dataset": "online_retail_II.csv",
    })

    # Cache last features for each SKU
    for sku, features in sku_last_features.items():
        model_service.cache_features(sku, features)

    print(f"\n  Model saved to: {model_service.model_dir / 'lightgbm_demand_forecast.pkl'}")
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
