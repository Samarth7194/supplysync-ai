# src/inventory/reorder_point.py

import numpy as np
from typing import Dict, Optional, Tuple
from src.uncertainty.dynamic_sigma import compute_dynamic_safety_stock, compute_rolling_forecast_error

def compute_reorder_decision(
    sku: str,
    current_stock: int,
    forecast: list,
    sigma: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    dynamic_safety_stock: Optional[float] = None,
    demand_variance: Optional[float] = None
) -> Dict:
    """
    Computes reorder decision for a single SKU with dynamic safety stock.
    
    Parameters:
    - sku: SKU identifier
    - current_stock: current inventory level
    - forecast: list of forecasted daily demand
    - sigma: standard deviation of forecast errors (fallback if dynamic not available)
    - lead_time_days: supplier lead time
    - service_level: desired service level (default 95%)
    - dynamic_safety_stock: pre-computed dynamic safety stock (optional)
    - demand_variance: demand variance for uncertainty calculations (optional)
    
    Returns:
    - Dictionary containing reorder decision details with explainability
    """
    
    # Z-score for service level
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    Z = z_scores.get(service_level, 1.65)
    
    # Expected demand during lead time
    lead_time_demand = sum(forecast[:lead_time_days])
    
    # Safety stock calculation (dynamic if available, otherwise fallback)
    if dynamic_safety_stock is not None:
        safety_stock = dynamic_safety_stock
        safety_stock_method = "dynamic"
    else:
        # Fallback to traditional method
        safety_stock = Z * sigma * np.sqrt(lead_time_days)
        safety_stock_method = "traditional"
    
    # Reorder point
    reorder_point = lead_time_demand + safety_stock
    
    # Quantity to order
    order_qty = max(0, reorder_point - current_stock)
    
    return {
        "sku": sku,
        "current_stock": current_stock,
        "lead_time_days": lead_time_days,
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "safety_stock_method": safety_stock_method,
        "reorder_point": round(reorder_point, 2),
        "order_quantity": int(round(order_qty)),
        "action": "REORDER" if order_qty > 0 else "NO_ACTION",
        "service_level": service_level,
        "z_score": Z,
        "explainability": {
            "lead_time_demand_calculation": f"Sum of forecast for {lead_time_days} days",
            "safety_stock_calculation": f"Z({Z}) × σ × √{lead_time_days}" if safety_stock_method == "traditional" else "Dynamic based on rolling forecast error",
            "reorder_point_calculation": "Lead Time Demand + Safety Stock",
            "decision_logic": "Order if Current Stock < Reorder Point"
        }
    }
