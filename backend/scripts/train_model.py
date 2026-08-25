"""
Train a LightGBM demand forecasting model on a retail transactions CSV.

Defaults to the UCI Online Retail II dataset. Supports bring-your-own-data via
``DATA_CSV_PATH`` (env var or ``--csv``), with optional ``--column-mapping``
for non-UCI column names. See ``docs/bring-your-own-data.md``.

Usage:
    cd backend
    python scripts/train_model.py
    DATA_CSV_PATH=/path/to/my.csv python scripts/train_model.py
    python scripts/train_model.py --csv my.csv --column-mapping cols.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_schema_checksum
from services.model_service import ModelService


@dataclass(frozen=True)
class TrainingPipelineResult:
    model_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    train_rows: int
    test_rows: int
    daily_rows: int
    sku_count: int


def prepare_sku_features(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Create lag + time features for a single SKU's demand data."""
    df = create_lag_features(sku_df, target_col="demand")
    df = create_time_features(df, date_col="date")
    df = df.dropna(subset=FEATURE_COLUMNS)
    return df


def train_lightgbm_demand_model(
    *,
    csv_path=None,
    column_mapping=None,
    parquet_path=None,
    model_dir: str | Path | None = None,
    artifact_file: str | None = None,
    metadata_file: str | None = None,
    verbose: bool = True,
) -> TrainingPipelineResult:
    """Run the existing temporal LightGBM training pipeline and save an artifact.

    This is the callable form used by controlled candidate training. It reuses
    the same feature engineering, temporal holdout, LightGBM config, metadata,
    checksum, and feature-schema behavior as the historical CLI.
    """

    def log(message: str = "") -> None:
        if verbose:
            print(message)

    log("=" * 60)
    log("SupplySync AI - Model Training Pipeline")
    log("=" * 60)

    if csv_path:
        log(f"Data source override: {csv_path}")
    elif os.environ.get("DATA_CSV_PATH"):
        log(f"Data source from env: {os.environ['DATA_CSV_PATH']}")

    log("\n[1/6] Loading and cleaning raw data...")
    raw_df = load_and_clean_retail_data(csv_path=csv_path, column_mapping=column_mapping)
    log(f"  Cleaned data: {len(raw_df):,} rows")

    log("[2/6] Aggregating to daily demand...")
    resolved_parquet = parquet_path or os.environ.get("DATA_PARQUET_PATH")
    daily_df = aggregate_daily_demand(raw_df, output_path=resolved_parquet)
    log(f"  Daily demand records: {len(daily_df):,}")
    log(f"  Unique SKUs: {daily_df['StockCode'].nunique():,}")

    log("[3/6] Selecting top SKUs...")
    top_skus = get_top_skus(daily_df, min_days=60, top_n=20)
    log(f"  Training on {len(top_skus)} SKUs: {top_skus[:5]}...")

    log("[4/6] Engineering features...")
    train_frames = []
    test_frames = []
    sku_last_features = {}

    for sku in top_skus:
        sku_df = load_sku_demand(sku, parquet_path=resolved_parquet)
        if len(sku_df) < 45:
            continue

        featured = prepare_sku_features(sku_df)
        if len(featured) < 30:
            continue

        split_idx = len(featured) - 30
        train_frames.append(featured.iloc[:split_idx])
        test_frames.append(featured.iloc[split_idx:])

        last_row = featured[FEATURE_COLUMNS].iloc[[-1]]
        sku_last_features[sku] = last_row

    if not train_frames or not test_frames:
        raise ValueError("Insufficient training data after feature engineering; need SKUs with usable temporal holdout windows.")

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    log(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    log("[5/6] Training LightGBM model...")
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

    preds = np.maximum(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    non_zero = y_test > 0
    mape = np.mean(np.abs((y_test[non_zero] - preds[non_zero]) / y_test[non_zero])) * 100 if non_zero.any() else 0

    log(f"  MAE:  {mae:.2f}")
    log(f"  RMSE: {rmse:.2f}")
    log(f"  MAPE: {mape:.1f}%")

    log("[6/6] Saving model and caching features...")
    saved_models_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent / "saved_models"
    model_service = ModelService(model_dir=str(saved_models_dir))

    dataset_label = "online_retail_II.csv"
    if csv_path:
        dataset_label = Path(csv_path).name
    elif os.environ.get("DATA_CSV_PATH"):
        dataset_label = Path(os.environ["DATA_CSV_PATH"]).name

    metadata = {
        "features": FEATURE_COLUMNS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_checksum": feature_schema_checksum(FEATURE_COLUMNS),
        "train_skus": top_skus,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 1),
        "dataset": dataset_label,
        "artifact_file": artifact_file or "lightgbm_demand_forecast.pkl",
        "metadata_file": metadata_file or "lightgbm_demand_forecast_metadata.json",
        "training_data": {
            "row_count": int(len(daily_df)),
            "sku_count": int(daily_df["StockCode"].nunique()),
            "date_start": str(daily_df["date"].min().date()) if "date" in daily_df else None,
            "date_end": str(daily_df["date"].max().date()) if "date" in daily_df else None,
        },
        "training_config": {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    }
    model_service.save_model(model, "lightgbm_demand_forecast", metadata=metadata)
    metadata = model_service.get_model_metadata("lightgbm_demand_forecast")

    for sku, features in sku_last_features.items():
        model_service.cache_features(sku, features)

    model_path = model_service.artifact_path("lightgbm_demand_forecast", metadata)
    metadata_path = model_service.model_dir / str(metadata.get("metadata_file", "lightgbm_demand_forecast_metadata.json"))
    log(f"\n  Model saved to: {model_path}")
    log("=" * 60)
    log("Training complete!")
    log("=" * 60)

    return TrainingPipelineResult(
        model_path=model_path,
        metadata_path=metadata_path,
        metadata=metadata,
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        daily_rows=int(len(daily_df)),
        sku_count=int(daily_df["StockCode"].nunique()),
    )


def train(csv_path=None, column_mapping=None, parquet_path=None, register_candidate=False):
    result = train_lightgbm_demand_model(
        csv_path=csv_path,
        column_mapping=column_mapping,
        parquet_path=parquet_path,
        verbose=True,
    )

    if register_candidate:
        from db.session import SessionLocal
        from repositories.model_artifact_repository import ModelArtifactRepository

        with SessionLocal() as session:
            artifact = ModelArtifactRepository(session).register_metadata(
                result.metadata,
                status="candidate",
            )
            session.commit()
        print(f"  Registered candidate model artifact id={artifact.id} version={artifact.version}")

    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the LightGBM demand forecaster.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a retail-transactions CSV. Defaults to DATA_CSV_PATH env var, then UCI Online Retail II.",
    )
    parser.add_argument(
        "--column-mapping",
        type=str,
        default=None,
        help="Path to a JSON file mapping {source_col: canonical_col} - required if your CSV doesn't use UCI column names.",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Override output parquet path. Defaults to DATA_PARQUET_PATH env var, then data/processed/daily_demand.parquet.",
    )
    parser.add_argument(
        "--register-candidate",
        action="store_true",
        help="Register the saved artifact in model_artifacts with candidate lifecycle status.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mapping = None
    if args.column_mapping:
        with open(args.column_mapping) as fh:
            mapping = json.load(fh)
    train(
        csv_path=args.csv,
        column_mapping=mapping,
        parquet_path=args.parquet,
        register_candidate=args.register_candidate,
    )
