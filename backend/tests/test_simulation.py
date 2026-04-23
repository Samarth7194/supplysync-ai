"""Tests for inventory simulation engine."""

import pytest
import pandas as pd
import numpy as np
from simulation.inventory_simulator import simulate_inventory


def _make_sku_df(demands):
    """Helper to create a sku_df from a list of demand values."""
    return pd.DataFrame({"demand": demands})


def test_no_demand_no_stockouts():
    df = _make_sku_df([0] * 30)
    result = simulate_inventory(
        sku_df=df, policy_fn=lambda inv: 0,
        initial_inventory=100, lead_time_days=7,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0
    )
    assert result["stockouts"] == 0
    assert result["stockout_cost"] == 0
    assert result["holding_cost"] > 0  # Still holding inventory


def test_depleting_inventory_causes_stockouts():
    df = _make_sku_df([10] * 30)
    result = simulate_inventory(
        sku_df=df, policy_fn=lambda inv: 0,  # Never reorder
        initial_inventory=50, lead_time_days=7,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0
    )
    assert result["stockouts"] > 0
    assert result["stockout_cost"] > 0


def test_reorder_policy_prevents_most_stockouts():
    df = _make_sku_df([5] * 60)
    # Reorder 50 when below 20
    result = simulate_inventory(
        sku_df=df, policy_fn=lambda inv: 50 if inv < 20 else 0,
        initial_inventory=100, lead_time_days=3,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0
    )
    # With reorder policy, stockouts should be much lower
    no_reorder = simulate_inventory(
        sku_df=df, policy_fn=lambda inv: 0,
        initial_inventory=100, lead_time_days=3,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0
    )
    assert result["stockouts"] < no_reorder["stockouts"]


def test_total_cost_is_sum_of_components():
    df = _make_sku_df([8] * 20)
    result = simulate_inventory(
        sku_df=df, policy_fn=lambda inv: 0,
        initial_inventory=50, lead_time_days=7,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0
    )
    assert abs(result["total_cost"] - (result["holding_cost"] + result["stockout_cost"])) < 0.01


def test_stockout_count_exact():
    # 5 days of demand=10, inventory=30, no reorder
    # Day 0: 30-10=20, Day 1: 20-10=10, Day 2: 10-10=0, Day 3: stockout 10, Day 4: stockout 10
    df = _make_sku_df([10] * 5)
    result = simulate_inventory(
        sku_df=df, policy_fn=lambda inv: 0,
        initial_inventory=30, lead_time_days=7,
        holding_cost_per_unit=0, stockout_cost_per_unit=1.0
    )
    assert result["stockouts"] == 20  # 10 + 10 unmet on days 3 and 4
    assert result["stockout_cost"] == 20.0
