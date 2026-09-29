"""Tests for the enhanced inventory simulator: inventory-position-based
policy decisions, the warm-up period, and the moving-average reorder-point
baseline.
"""

import pandas as pd
import pytest

from simulation.enhanced_simulator import EnhancedInventorySimulator, reweight_cost


def _make_sku_df(demands):
    return pd.DataFrame({"demand": demands})


def test_naive_policy_does_not_reorder_on_top_of_inbound_stock():
    """A policy deciding off on-hand alone (ignoring the pipeline) would keep
    re-ordering every day inventory stays below the threshold, even though a
    shipment is already inbound. Deciding off inventory *position* must stop
    that: once one order is placed, inventory position should clear the
    threshold and no further orders should fire before it arrives."""
    df = _make_sku_df([5] * 30)
    sim = EnhancedInventorySimulator(holding_cost_per_unit=0.1, stockout_cost_per_unit=1.0)
    result = sim.simulate_policy(
        sku_df=df,
        policy_name="naive",
        lead_time_days=7,
        initial_inventory=10,
        reorder_threshold=20,
        reorder_qty=1000,  # far more than 30 days of demand can deplete
    )
    # Only one reorder should fire: the first order (qty=1000) immediately
    # pushes inventory position (10 + 1000) far above the threshold, and
    # stays there for the rest of the window, so no further orders should be
    # placed while it's in transit or after it arrives.
    assert len(result.reorder_events) == 1


def test_warmup_days_are_excluded_from_every_metric():
    """Demand during the warm-up window must not count toward holding cost,
    stockout cost, fill rate, or the measured-day count."""
    # First 10 days: heavy demand causing stockouts. Remaining 20: no demand.
    demands = [50] * 10 + [0] * 20
    df = _make_sku_df(demands)
    sim = EnhancedInventorySimulator(holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0)

    warm = sim.simulate_policy(
        sku_df=df, policy_name="naive", initial_inventory=0,
        warmup_days=10, reorder_threshold=0, reorder_qty=0,
    )
    no_warm = sim.simulate_policy(
        sku_df=df, policy_name="naive", initial_inventory=0,
        warmup_days=0, reorder_threshold=0, reorder_qty=0,
    )

    assert warm.measured_days == 20
    assert warm.stockouts == 0  # all stockouts happened during warm-up
    assert no_warm.stockouts > 0  # without warm-up, the same stockouts are counted
    assert len(warm.inventory_timeseries) == 20


def test_warmup_days_must_be_shorter_than_the_series():
    df = _make_sku_df([1] * 5)
    sim = EnhancedInventorySimulator()
    with pytest.raises(ValueError):
        sim.simulate_policy(sku_df=df, policy_name="naive", warmup_days=5)


def test_moving_average_policy_reorders_below_the_reorder_point():
    sim = EnhancedInventorySimulator()
    recent_demand = pd.Series([10.0] * 10)
    # avg_demand=10, lead_time=7 -> reorder_point = 70 + safety_stock
    order = sim.moving_average_policy(
        inventory_position=50, recent_demand=recent_demand,
        lead_time_days=7, safety_stock=20, review_period_days=7,
    )
    assert order > 0
    # order-up-to level = 10*(7+7) + 20 = 160; order = 160 - 50 = 110
    assert order == 110


def test_moving_average_policy_does_not_reorder_above_the_reorder_point():
    sim = EnhancedInventorySimulator()
    recent_demand = pd.Series([10.0] * 10)
    order = sim.moving_average_policy(
        inventory_position=200, recent_demand=recent_demand,
        lead_time_days=7, safety_stock=20, review_period_days=7,
    )
    assert order == 0


def test_moving_average_rop_integrates_through_simulate_policy():
    df = _make_sku_df([8] * 60)
    sim = EnhancedInventorySimulator(holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0)
    result = sim.simulate_policy(
        sku_df=df, policy_name="moving_average_rop", lead_time_days=7,
        warmup_days=10, safety_stock=15.0,
    )
    assert result.measured_days == 50
    assert result.fill_rate > 0.9  # a sane ROP policy should avoid most stockouts here


def test_reweight_cost_matches_a_fresh_simulation_at_the_same_ratio():
    """reweight_cost must reproduce simulate_policy's own totals when given
    back the same cost-per-unit values it was originally run with, since no
    policy here takes cost parameters as input."""
    df = _make_sku_df([6, 0, 9, 3, 12, 0, 4] * 8)
    sim = EnhancedInventorySimulator(holding_cost_per_unit=0.5, stockout_cost_per_unit=5.0)
    result = sim.simulate_policy(
        sku_df=df, policy_name="naive", lead_time_days=7,
        warmup_days=5, reorder_threshold=15, reorder_qty=40,
    )
    holding, stockout, total = reweight_cost(result, 0.5, 5.0)
    assert holding == pytest.approx(result.holding_cost)
    assert stockout == pytest.approx(result.stockout_cost)
    assert total == pytest.approx(result.total_cost)


def test_reweight_cost_changes_with_a_different_ratio():
    df = _make_sku_df([10] * 30)
    sim = EnhancedInventorySimulator(holding_cost_per_unit=1.0, stockout_cost_per_unit=1.0)
    result = sim.simulate_policy(
        sku_df=df, policy_name="naive", lead_time_days=7,
        warmup_days=5, reorder_threshold=0, reorder_qty=0,  # never reorder -> guaranteed stockouts
    )
    assert result.stockouts > 0
    _, _, cheap_stockout_total = reweight_cost(result, holding_cost_per_unit=1.0, stockout_cost_per_unit=1.0)
    _, _, expensive_stockout_total = reweight_cost(result, holding_cost_per_unit=1.0, stockout_cost_per_unit=100.0)
    assert expensive_stockout_total > cheap_stockout_total
