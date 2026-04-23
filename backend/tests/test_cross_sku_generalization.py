"""Tests for bring-your-own-data column mapping + cross-SKU generalization.

These tests focus on the plumbing — they verify the column-mapping logic,
graceful failure on missing columns, and that the DATA_CSV_PATH env var is
honored. The cross-SKU evaluation script itself is expensive (retrains
LightGBM multiple times) and is covered by a lightweight smoke test that
only runs when the processed parquet + model artifact are both present.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from ingestion.load_retail_data import load_and_clean_retail_data


BACKEND_DIR = Path(__file__).resolve().parent.parent


def _write_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_column_mapping_renames_to_canonical(tmp_path: Path) -> None:
    """A CSV with non-UCI column names loads cleanly when a mapping is provided."""
    csv = tmp_path / "my_sales.csv"
    _write_csv(csv, [
        {"transaction_date": "2024-01-01", "units_sold": 10, "product_id": "SKUA", "unit_price": 1.50},
        {"transaction_date": "2024-01-02", "units_sold": 5, "product_id": "SKUA", "unit_price": 1.50},
        {"transaction_date": "2024-01-03", "units_sold": 20, "product_id": "SKUB", "unit_price": 2.00},
    ])

    mapping = {
        "transaction_date": "InvoiceDate",
        "units_sold": "Quantity",
        "product_id": "StockCode",
        "unit_price": "Price",
    }
    df = load_and_clean_retail_data(csv_path=csv, column_mapping=mapping)

    assert "InvoiceDate" in df.columns
    assert "Quantity" in df.columns
    assert "StockCode" in df.columns
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])
    assert set(df["StockCode"].unique()) == {"SKUA", "SKUB"}


def test_missing_required_columns_raises(tmp_path: Path) -> None:
    """Without a usable mapping and missing canonical columns, loader raises."""
    csv = tmp_path / "bad.csv"
    _write_csv(csv, [
        {"foo": "2024-01-01", "bar": 10, "baz": "X"},
    ])

    with pytest.raises(ValueError) as exc_info:
        load_and_clean_retail_data(csv_path=csv)

    msg = str(exc_info.value)
    assert "missing required columns" in msg
    assert "InvoiceDate" in msg or "Quantity" in msg or "StockCode" in msg


def test_env_var_csv_path_is_honored(tmp_path: Path, monkeypatch) -> None:
    """When csv_path is None, DATA_CSV_PATH env var wins over the default."""
    csv = tmp_path / "env_source.csv"
    _write_csv(csv, [
        {"InvoiceDate": "2024-02-01", "Quantity": 3, "StockCode": "ENVSKU", "Price": 1.0},
        {"InvoiceDate": "2024-02-02", "Quantity": 4, "StockCode": "ENVSKU", "Price": 1.0},
    ])
    monkeypatch.setenv("DATA_CSV_PATH", str(csv))
    df = load_and_clean_retail_data()
    assert (df["StockCode"] == "ENVSKU").all()
    assert len(df) == 2


def test_partial_column_mapping_is_accepted(tmp_path: Path) -> None:
    """A mapping that only remaps some columns works — the rest use canonical names as-is."""
    csv = tmp_path / "partial.csv"
    _write_csv(csv, [
        # Date is non-canonical; quantity + stock code already match UCI names.
        {"sale_date": "2024-03-01", "Quantity": 10, "StockCode": "P1", "Price": 1.0},
        {"sale_date": "2024-03-02", "Quantity": 12, "StockCode": "P1", "Price": 1.0},
    ])
    df = load_and_clean_retail_data(csv_path=csv, column_mapping={"sale_date": "InvoiceDate"})
    assert "InvoiceDate" in df.columns
    assert len(df) == 2


def test_negative_quantities_are_dropped(tmp_path: Path) -> None:
    """Returns (negative quantities) are filtered during cleaning."""
    csv = tmp_path / "returns.csv"
    _write_csv(csv, [
        {"InvoiceDate": "2024-04-01", "Quantity": 10, "StockCode": "A", "Price": 1.0},
        {"InvoiceDate": "2024-04-02", "Quantity": -5, "StockCode": "A", "Price": 1.0},
        {"InvoiceDate": "2024-04-03", "Quantity": 7, "StockCode": "A", "Price": 1.0},
    ])
    df = load_and_clean_retail_data(csv_path=csv)
    assert (df["Quantity"] > 0).all()
    assert len(df) == 2


@pytest.mark.skipif(
    not (BACKEND_DIR.parent / "data" / "processed" / "daily_demand.parquet").exists(),
    reason="Requires processed parquet from `make bootstrap`.",
)
@pytest.mark.skipif(
    not (BACKEND_DIR / "saved_models" / "lightgbm_demand_forecast.pkl").exists(),
    reason="Requires trained LightGBM artifact from `make bootstrap`.",
)
def test_cross_sku_script_produces_valid_output(tmp_path: Path) -> None:
    """Smoke-test: the cross-SKU script runs end-to-end and produces a valid JSON schema.

    Uses --folds 2 --top-skus 4 to keep the run short.
    """
    output = tmp_path / "cross.json"
    result = subprocess.run(
        [
            sys.executable,
            str(BACKEND_DIR / "scripts" / "evaluate_cross_sku.py"),
            "--folds", "2",
            "--top-skus", "4",
            "--output", str(output),
        ],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output.exists(), "expected JSON report at --output"

    payload = json.loads(output.read_text())
    for key in ("generated_at", "folds", "top_skus", "aggregate", "winner_rate", "fold_summaries", "per_sku"):
        assert key in payload, f"missing key: {key}"
    assert payload["folds"] == 2
    assert payload["top_skus"] == 4
    assert isinstance(payload["per_sku"], list)
    # Each fold assigned at least one held-out SKU (2 folds × 4 SKUs = 2 per fold).
    assert all(len(fs["holdout_skus"]) >= 1 for fs in payload["fold_summaries"])
