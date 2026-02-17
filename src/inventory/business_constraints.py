# src/inventory/business_constraints.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SupplierConstraints:
    """Business constraints for a specific supplier/SKU."""
    moq: int = 1  # Minimum Order Quantity
    order_multiple: int = 1  # Order in multiples of this number
    max_order_quantity: Optional[int] = None  # Maximum order per batch
    lead_time_days: int = 7
    
@dataclass
class BudgetConstraints:
    """Budget constraints across SKUs."""
    monthly_budget: float = 10000.0
    budget_utilization_pct: float = 0.8  # Use only 80% of budget for safety
    cost_per_unit: Dict[str, float] = None  # Cost per unit by SKU
    
    def __post_init__(self):
        if self.cost_per_unit is None:
            self.cost_per_unit = {}

def apply_moq_constraint(order_quantity: int, moq: int) -> int:
    """
    Apply Minimum Order Quantity constraint.
    
    Parameters:
    - order_quantity: Calculated order quantity
    - moq: Minimum order quantity required
    
    Returns:
    - Adjusted order quantity respecting MOQ
    """
    
    if order_quantity == 0:
        return 0
    
    return max(order_quantity, moq)

def apply_order_multiple_constraint(order_quantity: int, order_multiple: int) -> int:
    """
    Apply order multiple constraint.
    
    Parameters:
    - order_quantity: Calculated order quantity
    - order_multiple: Must order in multiples of this
    
    Returns:
    - Adjusted order quantity respecting multiples
    """
    
    if order_quantity == 0 or order_multiple == 1:
        return order_quantity
    
    # Round up to nearest multiple
    adjusted_quantity = ((order_quantity + order_multiple - 1) // order_multiple) * order_multiple
    
    return adjusted_quantity

def apply_max_order_constraint(order_quantity: int, max_order: Optional[int]) -> int:
    """
    Apply maximum order quantity constraint.
    
    Parameters:
    - order_quantity: Calculated order quantity
    - max_order: Maximum allowed order quantity
    
    Returns:
    - Adjusted order quantity respecting maximum
    """
    
    if max_order is None or order_quantity <= max_order:
        return order_quantity
    
    # If order exceeds maximum, cap it and flag for manual review
    return max_order

def optimize_across_budget(
    decisions: Dict[str, Dict],
    budget_constraints: BudgetConstraints,
    supplier_constraints: Dict[str, SupplierConstraints] = None
) -> Dict[str, Dict]:
    """
    Optimize order quantities across SKUs to respect budget constraints.
    
    Parameters:
    - decisions: Reorder decisions by SKU
    - budget_constraints: Budget limitations
    - supplier_constraints: Supplier constraints by SKU
    
    Returns:
    - Budget-optimized decisions
    """
    
    if supplier_constraints is None:
        supplier_constraints = {}
    
    # Calculate total cost of recommended orders
    total_cost = 0
    sku_costs = {}
    
    for sku, decision in decisions.items():
        if decision.get("action") == "REORDER":
            cost_per_unit = budget_constraints.cost_per_unit.get(sku, 1.0)
            order_qty = decision["order_quantity"]
            cost = order_qty * cost_per_unit
            
            sku_costs[sku] = cost
            total_cost += cost
    
    # Check if budget is exceeded
    available_budget = budget_constraints.monthly_budget * budget_constraints.budget_utilization_pct
    
    if total_cost <= available_budget:
        # No budget adjustment needed
        return decisions
    
    # Budget exceeded - need to optimize
    # Strategy: Prioritize by criticality (safety stock ratio)
    
    # Calculate priority scores
    sku_priorities = {}
    for sku, decision in decisions.items():
        if decision.get("action") == "REORDER":
            # Higher priority for higher safety stock ratio
            safety_stock_ratio = decision["safety_stock"] / decision["reorder_point"]
            urgency = 1 - (decision["current_stock"] / decision["reorder_point"])
            
            priority_score = safety_stock_ratio * urgency
            sku_priorities[sku] = priority_score
    
    # Sort by priority (highest first)
    sorted_skus = sorted(sku_priorities.keys(), key=lambda x: sku_priorities[x], reverse=True)
    
    # Allocate budget greedily by priority
    remaining_budget = available_budget
    optimized_decisions = decisions.copy()
    
    for sku in sorted_skus:
        if remaining_budget <= 0:
            # Budget exhausted - cancel remaining orders
            optimized_decisions[sku]["order_quantity"] = 0
            optimized_decisions[sku]["action"] = "NO_ACTION"
            optimized_decisions[sku]["budget_constraint"] = "cancelled_budget_exhausted"
            continue
        
        cost_per_unit = budget_constraints.cost_per_unit.get(sku, 1.0)
        original_qty = decisions[sku]["order_quantity"]
        original_cost = original_qty * cost_per_unit
        
        if original_cost <= remaining_budget:
            # Full order fits in budget
            remaining_budget -= original_cost
            optimized_decisions[sku]["budget_constraint"] = "full_order_approved"
        else:
            # Partial order only
            max_affordable_qty = int(remaining_budget // cost_per_unit)
            
            # Apply supplier constraints to partial quantity
            if sku in supplier_constraints:
                supplier = supplier_constraints[sku]
                max_affordable_qty = apply_moq_constraint(max_affordable_qty, supplier.moq)
                max_affordable_qty = apply_order_multiple_constraint(max_affordable_qty, supplier.order_multiple)
            
            if max_affordable_qty > 0:
                optimized_decisions[sku]["order_quantity"] = max_affordable_qty
                optimized_decisions[sku]["action"] = "REORDER"
                optimized_decisions[sku]["budget_constraint"] = f"partial_order_{max_affordable_qty}/{original_qty}"
                remaining_budget -= max_affordable_qty * cost_per_unit
            else:
                optimized_decisions[sku]["order_quantity"] = 0
                optimized_decisions[sku]["action"] = "NO_ACTION"
                optimized_decisions[sku]["budget_constraint"] = "cancelled_insufficient_budget"
    
    return optimized_decisions

def apply_business_constraints(
    decision: Dict,
    supplier_constraints: SupplierConstraints
) -> Dict:
    """
    Apply all business constraints to a single reorder decision.
    
    Parameters:
    - decision: Base reorder decision
    - supplier_constraints: Business constraints
    
    Returns:
    - Constraint-adjusted decision
    """
    
    if decision["action"] == "NO_ACTION":
        return decision
    
    original_quantity = decision["order_quantity"]
    adjusted_quantity = original_quantity
    
    # Apply constraints in sequence
    adjusted_quantity = apply_moq_constraint(adjusted_quantity, supplier_constraints.moq)
    adjusted_quantity = apply_order_multiple_constraint(adjusted_quantity, supplier_constraints.order_multiple)
    adjusted_quantity = apply_max_order_constraint(adjusted_quantity, supplier_constraints.max_order_quantity)
    
    # Update decision
    constrained_decision = decision.copy()
    constrained_decision["order_quantity"] = adjusted_quantity
    constrained_decision["action"] = "REORDER" if adjusted_quantity > 0 else "NO_ACTION"
    
    # Add constraint metadata
    constraints_applied = []
    
    if adjusted_quantity != original_quantity:
        if adjusted_quantity >= supplier_constraints.moq and original_quantity < supplier_constraints.moq:
            constraints_applied.append(f"MOQ increased from {original_quantity} to {adjusted_quantity}")
        
        if supplier_constraints.order_multiple > 1:
            if adjusted_quantity % supplier_constraints.order_multiple == 0 and original_quantity % supplier_constraints.order_multiple != 0:
                constraints_applied.append(f"Rounded to multiple of {supplier_constraints.order_multiple}")
        
        if supplier_constraints.max_order_quantity and adjusted_quantity == supplier_constraints.max_order_quantity:
            constraints_applied.append(f"Capped at maximum {supplier_constraints.max_order_quantity}")
    
    constrained_decision["business_constraints"] = {
        "moq": supplier_constraints.moq,
        "order_multiple": supplier_constraints.order_multiple,
        "max_order_quantity": supplier_constraints.max_order_quantity,
        "constraints_applied": constraints_applied,
        "original_quantity": original_quantity,
        "final_quantity": adjusted_quantity
    }
    
    return constrained_decision

def get_default_supplier_constraints() -> Dict[str, SupplierConstraints]:
    """
    Get default supplier constraints for common SKU categories.
    
    Returns:
    - Dictionary of default constraints by SKU pattern
    """
    
    defaults = {
        "regular": SupplierConstraints(moq=10, order_multiple=5, max_order_quantity=1000),
        "intermittent": SupplierConstraints(moq=5, order_multiple=1, max_order_quantity=500),
        "highly_intermittent": SupplierConstraints(moq=1, order_multiple=1, max_order_quantity=100)
    }
    
    return defaults
