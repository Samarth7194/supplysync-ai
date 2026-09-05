"""Data ingestion pipeline for Online Retail II (and compatible) datasets.

By default the loader expects the UCI Online Retail II column names
(``InvoiceDate``, ``Quantity``, ``StockCode``, ``Price``, ``Description``).
For different schemas, pass a ``column_mapping`` dict or set the
``DATA_CSV_PATH`` env var pointing at a CSV that already uses UCI names.

See ``docs/bring-your-own-data.md`` for worked examples.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def _project_root() -> Path:
    """Find the project root — the directory containing ``data/raw/`` or
    ``data/processed/``.

    ``data/raw/`` holds the source CSV (may be absent after a clean clone —
    users download it separately). ``data/processed/`` holds the generated
    parquet. Either one is sufficient evidence that we've found the root.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data" / "raw").exists() or (parent / "data" / "processed").exists():
            return parent
    cwd = Path.cwd()
    for candidate in (cwd, cwd.parent):
        if (candidate / "data" / "raw").exists() or (candidate / "data" / "processed").exists():
            return candidate
    return cwd


def _resolve_csv_path(csv_path: Optional[PathLike]) -> Path:
    """Resolve the CSV path, honoring explicit argument then ``DATA_CSV_PATH``."""
    if csv_path is not None:
        return Path(csv_path)
    env_override = os.environ.get("DATA_CSV_PATH")
    if env_override:
        return Path(env_override)
    return _project_root() / "data" / "raw" / "online_retail_II.csv"


def _apply_column_mapping(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """Rename user-side columns to the canonical UCI names, if needed.

    ``column_mapping`` is a {source_name: canonical_name} dict. Keys that
    aren't present in ``df.columns`` are silently ignored so partial mappings
    work (e.g. user only needs to remap the date column).
    """
    if not column_mapping:
        return df
    rename_map = {src: dst for src, dst in column_mapping.items() if src in df.columns}
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def load_and_clean_retail_data(
    csv_path: Optional[PathLike] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load and clean a retail-transactions CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file. Defaults to ``DATA_CSV_PATH`` env var, then to
        ``data/raw/online_retail_II.csv`` under the project root.
    column_mapping:
        Optional ``{source_name: canonical_name}`` dict. Use when the CSV
        uses column names different from the UCI canonical set
        (``InvoiceDate``, ``Quantity``, ``StockCode``, ``Price``,
        ``Description``). Partial mappings are fine.
    """
    csv_path = _resolve_csv_path(csv_path)

    # Read without parse_dates first because after renaming the date column
    # may live under a non-default name.
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    df = _apply_column_mapping(df, column_mapping)

    missing_required = [c for c in ("InvoiceDate", "Quantity", "StockCode") if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"CSV at {csv_path} is missing required columns after mapping: {missing_required}. "
            f"Columns present: {list(df.columns)}. Pass column_mapping= to rename them."
        )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Drop cancellations and returns (negative quantities)
    df = df[df["Quantity"] > 0]

    # Drop invalid prices — only if the column exists in the mapped dataframe
    if "Price" in df.columns:
        df = df[df["Price"] > 0]

    # Drop null StockCodes
    df = df.dropna(subset=["StockCode"])

    # Remove service/non-product codes
    service_codes = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "C2", "AMAZONFEE"}
    df = df[~df["StockCode"].astype(str).str.upper().isin(service_codes)]

    # Keep only alphanumeric stock codes (filter out test/adjustment entries)
    df = df[df["StockCode"].astype(str).str.match(r"^[A-Za-z0-9]+$")]

    return df


def aggregate_daily_demand(df: pd.DataFrame, output_path: Optional[PathLike] = None) -> pd.DataFrame:
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


def load_sku_demand(sku: str, parquet_path: Optional[PathLike] = None) -> pd.DataFrame:
    if parquet_path is None:
        env_override = os.environ.get("DATA_PARQUET_PATH")
        if env_override:
            parquet_path = env_override
        else:
            parquet_path = _project_root() / "data" / "processed" / "daily_demand.parquet"

    daily = pd.read_parquet(parquet_path)
    sku_data = daily[daily["StockCode"] == sku].copy()

    if len(sku_data) == 0:
        return pd.DataFrame(columns=["date", "demand"])

    sku_data = sku_data.sort_values("date")
    full_range = pd.date_range(sku_data["date"].min(), sku_data["date"].max(), freq="D")
    # Only the demand column needs continuous coverage; dropping StockCode before
    # reindex avoids pandas 2.x strict-string dtypes rejecting `fill_value=0`.
    demand_series = sku_data.set_index("date")["demand"].reindex(full_range, fill_value=0)
    demand_series.index.name = "date"

    return demand_series.reset_index().rename(columns={"index": "date"})[["date", "demand"]]


def get_sku_descriptions(
    csv_path: Optional[PathLike] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> dict:
    """Return a mapping of StockCode -> Description from the raw data."""
    csv_path = _resolve_csv_path(csv_path)

    df = pd.read_csv(csv_path, encoding="ISO-8859-1", nrows=500000)
    df = _apply_column_mapping(df, column_mapping)
    if "Description" not in df.columns or "StockCode" not in df.columns:
        return {}
    df = df[["StockCode", "Description"]].dropna(subset=["StockCode", "Description"])
    desc_map = df.groupby("StockCode")["Description"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    return desc_map.to_dict()
