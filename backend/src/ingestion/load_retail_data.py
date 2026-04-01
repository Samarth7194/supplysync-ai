"""Data ingestion pipeline for Online Retail II dataset."""

import pandas as pd
import numpy as np
from pathlib import Path


def _project_root() -> Path:
    """Find the project root (directory containing 'data/')."""
    # Walk up from this file to find the directory that contains 'data/raw/'
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data" / "raw").exists():
            return parent
    # Fallback: assume CWD is project root or backend/
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists():
        return cwd
    if (cwd.parent / "data" / "raw").exists():
        return cwd.parent
    return cwd


def load_and_clean_retail_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = _project_root() / "data" / "raw" / "online_retail_II.csv"

    df = pd.read_csv(csv_path, encoding="ISO-8859-1", parse_dates=["InvoiceDate"])

    # Drop cancellations and returns (negative quantities)
    df = df[df["Quantity"] > 0]

    # Drop invalid prices
    df = df[df["Price"] > 0]

    # Drop null StockCodes
    df = df.dropna(subset=["StockCode"])

    # Remove service/non-product codes
    service_codes = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "C2", "AMAZONFEE"}
    df = df[~df["StockCode"].astype(str).str.upper().isin(service_codes)]

    # Keep only alphanumeric stock codes (filter out test/adjustment entries)
    df = df[df["StockCode"].astype(str).str.match(r"^[A-Za-z0-9]+$")]

    return df


def aggregate_daily_demand(df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    if output_path is None:
        output_path = _project_root() / "data" / "processed" / "daily_demand.parquet"

    df = df.copy()
    df["date"] = df["InvoiceDate"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby(["StockCode", "date"])["Quantity"].sum().reset_index()
    daily.columns = ["StockCode", "date", "demand"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False)

    return daily


def get_top_skus(daily_demand_df: pd.DataFrame, min_days: int = 60, top_n: int = 20) -> list:
    sku_stats = daily_demand_df.groupby("StockCode").agg(
        n_days=("date", "nunique"),
        total_demand=("demand", "sum")
    ).reset_index()

    qualified = sku_stats[sku_stats["n_days"] >= min_days]
    top = qualified.nlargest(top_n, "total_demand")
    return top["StockCode"].tolist()


def load_sku_demand(sku: str, parquet_path: str = None) -> pd.DataFrame:
    if parquet_path is None:
        parquet_path = _project_root() / "data" / "processed" / "daily_demand.parquet"

    daily = pd.read_parquet(parquet_path)
    sku_data = daily[daily["StockCode"] == sku].copy()

    if len(sku_data) == 0:
        return pd.DataFrame(columns=["date", "demand"])

    sku_data = sku_data.sort_values("date")
    full_range = pd.date_range(sku_data["date"].min(), sku_data["date"].max(), freq="D")
    sku_data = sku_data.set_index("date").reindex(full_range, fill_value=0)
    sku_data.index.name = "date"
    sku_data = sku_data.reset_index()

    return sku_data[["date", "demand"]]


def get_sku_descriptions(csv_path: str = None) -> dict:
    """Return a mapping of StockCode -> Description from the raw data."""
    if csv_path is None:
        csv_path = _project_root() / "data" / "raw" / "online_retail_II.csv"

    df = pd.read_csv(csv_path, encoding="ISO-8859-1", usecols=["StockCode", "Description"], nrows=500000)
    df = df.dropna(subset=["StockCode", "Description"])
    desc_map = df.groupby("StockCode")["Description"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    return desc_map.to_dict()
