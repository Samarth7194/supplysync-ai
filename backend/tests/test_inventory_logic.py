"""Tests for core inventory logic: reorder decisions and business constraints."""

import pytest
from inventory.reorder_point import compute_reorder_decision
from inventory.business_constraints import (
    apply_moq_constraint,
    apply_order_multiple_constraint,
    apply_max_order_constraint,
    apply_business_constraints,
    optimize_across_budget,
    SupplierConstraints,
    BudgetConstraints,
)


# --- Reorder Decision Tests ---

def test_reorder_triggers_when_stock_low():
    decision = compute_reorder_decision(
        sku="TEST-001", current_stock=10,
        forecast=[5.0] * 7, sigma=2.0, lead_time_days=7
    )
    assert decision["action"] == "REORDER"
    assert decision["order_quantity"] > 0


def test_no_action_when_stock_high():
    decision = compute_reorder_decision(
        sku="TEST-002", current_stock=200,
        forecast=[5.0] * 7, sigma=2.0, lead_time_days=7
    )
    assert decision["action"] == "NO_ACTION"
    assert decision["order_quantity"] == 0


def test_dynamic_safety_stock_overrides_traditional():
    decision = compute_reorder_decision(
        sku="TEST-003", current_stock=10,
        forecast=[5.0] * 7, sigma=2.0, lead_time_days=7,
        dynamic_safety_stock=15.0
    )
    assert decision["safety_stock"] == 15.0
    assert decision["safety_stock_method"] == "dynamic"


def test_lead_time_demand_sums_forecast():
    forecast = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    decision = compute_reorder_decision(
        sku="TEST-004", current_stock=0,
        forecast=forecast, sigma=0.0, lead_time_days=7
    )
    assert decision["lead_time_demand"] == 280.0


def test_order_quantity_is_non_negative():
    decision = compute_reorder_decision(
        sku="TEST-005", current_stock=9999,
        forecast=[1.0] * 7, sigma=1.0, lead_time_days=7
    )
    assert decision["order_quantity"] >= 0


# --- Business Constraint Tests ---

def test_moq_increases_small_quantity():
    assert apply_moq_constraint(3, 10) == 10


def test_moq_zero_stays_zero():
    assert apply_moq_constraint(0, 10) == 0


def test_moq_large_quantity_unchanged():
    assert apply_moq_constraint(50, 10) == 50


def test_order_multiple_rounds_up():
    assert apply_order_multiple_constraint(13, 5) == 15


def test_order_multiple_exact_unchanged():
    assert apply_order_multiple_constraint(15, 5) == 15


def test_order_multiple_one_unchanged():
    assert apply_order_multiple_constraint(13, 1) == 13


def test_max_order_caps_quantity():
    assert apply_max_order_constraint(500, 200) == 200


def test_max_order_none_unchanged():
    assert apply_max_order_constraint(500, None) == 500


def test_apply_business_constraints_pipeline():
    decision = {
        "order_quantity": 3,
        "sku": "TEST",
        "action": "REORDER",
    }
    constraints = SupplierConstraints(moq=10, order_multiple=12, max_order_quantity=100)
    result = apply_business_constraints(decision, constraints)
    # MOQ bumps 3->10, then multiple rounds 10->12
    assert result["order_quantity"] == 12
    assert "business_constraints" in result
    assert len(result["business_constraints"]["constraints_applied"]) > 0


def test_budget_optimization_reduces_over_budget():
    decisions = {
        "SKU-A": {
            "sku": "SKU-A", "order_quantity": 100, "action": "REORDER",
            "safety_stock": 20, "reorder_point": 50, "current_stock": 10,
        },
        "SKU-B": {
            "sku": "SKU-B", "order_quantity": 100, "action": "REORDER",
            "safety_stock": 10, "reorder_point": 30, "current_stock": 5,
        },
    }
    budget = BudgetConstraints(
        monthly_budget=500.0, budget_utilization_pct=1.0,
        cost_per_unit={"SKU-A": 5.0, "SKU-B": 5.0}
    )
    result = optimize_across_budget(decisions, budget)
    total_cost = sum(
        r["order_quantity"] * budget.cost_per_unit.get(sku, 0)
        for sku, r in result.items()
    )
    assert total_cost <= 500.0
