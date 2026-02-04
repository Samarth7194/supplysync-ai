# src/inventory/reorder_engine.py

import numpy as np

def compute_reorder_decision(
    sku,
    current_stock,
    forecast,
    sigma,
    lead_time_days=7,
    service_level=0.95
):
    """
    Computes reorder decision for a single SKU.
    
    Parameters:
    - sku: SKU identifier
    - current_stock: current inventory level
    - forecast: list of forecasted daily demand
    - sigma: standard deviation of forecast errors
    - lead_time_days: supplier lead time
    - service_level: desired service level (default 95%)
    
    Returns:
    - Dictionary containing reorder decision details
    """
    
    # Z-score for service level (95%)
    Z = 1.65 if service_level == 0.95 else 1.28
    
    # Expected demand during lead time
    lead_time_demand = sum(forecast[:lead_time_days])
    
    # Safety stock to protect against uncertainty
    safety_stock = Z * sigma * np.sqrt(lead_time_days)
    
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
        "reorder_point": round(reorder_point, 2),
        "order_quantity": int(round(order_qty)),
        "action": "REORDER" if order_qty > 0 else "NO_ACTION"
    }
