"""Singleton data service for loading demand data from parquet."""

import pandas as pd
from pathlib import Path


class DataService:
    _instance = None
    _demand_data = None
    _sku_descriptions = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_data()
        return cls._instance

    def _find_parquet(self):
        """Find daily_demand.parquet by walking up from this file."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "data" / "processed" / "daily_demand.parquet"
            if candidate.exists():
                return candidate
        return None

    def _load_data(self):
        path = self._find_parquet()
        if path is None:
            raise FileNotFoundError(
                "daily_demand.parquet not found. Run: python scripts/train_model.py"
            )
        self._demand_data = pd.read_parquet(path)

    def get_demand_history(self, sku: str) -> pd.Series:
        df = self._demand_data[self._demand_data["StockCode"] == sku].copy()
        if len(df) == 0:
            return pd.Series(dtype=float)
        df = df.sort_values("date")
        full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        df = df.set_index("date").reindex(full_range, fill_value=0)["demand"]
        return df

    def get_available_skus(self) -> list:
        return self._demand_data["StockCode"].unique().tolist()

    def get_top_skus(self, n: int = 20) -> list:
        stats = self._demand_data.groupby("StockCode").agg(
            total=("demand", "sum"),
            days=("date", "nunique"),
        ).reset_index()
        qualified = stats[stats["days"] >= 30]
        return qualified.nlargest(n, "total")["StockCode"].tolist()
