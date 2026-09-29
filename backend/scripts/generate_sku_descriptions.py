"""
Generate backend/data/sku_descriptions.json: a small, committed mapping of
StockCode -> Description for the SKUs this deployment actually serves.

Why this exists
----------------
Product names come from the raw CSV (data/raw/online_retail_II.csv), which
is gitignored -- a fresh clone or a production deploy that only ships the
processed parquet has no raw CSV, so every SKU falls back to "SKU {code}"
in the UI. This script runs once (wherever the raw CSV happens to be
available) and commits a tiny JSON file covering the SKUs /api/skus
actually serves, so production shows real product names without needing
the raw CSV at all.

Usage:
    cd backend
    python scripts/generate_sku_descriptions.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from ingestion.load_retail_data import get_sku_descriptions

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
# A buffer over the 20 SKUs /api/skus currently serves (DataService.get_top_skus
# default n=20), so this file stays valid if that default changes slightly.
TOP_N = 50


def generate():
    parquet_path = os.path.join(BACKEND_DIR, "..", "data", "processed", "daily_demand.parquet")
    daily_df = pd.read_parquet(parquet_path)

    stats = daily_df.groupby("StockCode").agg(
        total=("demand", "sum"),
        days=("date", "nunique"),
    ).reset_index()
    qualified = stats[stats["days"] >= 30]  # matches DataService.get_top_skus
    top_skus = set(qualified.nlargest(TOP_N, "total")["StockCode"].tolist())

    all_descriptions = get_sku_descriptions()
    served_descriptions = {
        sku: desc for sku, desc in all_descriptions.items() if sku in top_skus
    }

    missing = top_skus - served_descriptions.keys()
    if missing:
        print(f"WARNING: no description found for {len(missing)} served SKUs: {sorted(missing)}")

    out_path = os.path.join(BACKEND_DIR, "data", "sku_descriptions.json")
    with open(out_path, "w") as f:
        json.dump(served_descriptions, f, indent=2, sort_keys=True)

    print(f"Wrote {len(served_descriptions)} descriptions (of {len(top_skus)} served SKUs) to {out_path}")


if __name__ == "__main__":
    generate()
