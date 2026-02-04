# src/simulation/inventory_simulator.py

def simulate_inventory(
    sku_df,
    policy_fn,
    initial_inventory,
    lead_time_days,
    holding_cost_per_unit,
    stockout_cost_per_unit
):
    """
    Simulates day-by-day inventory operations for a single SKU.
    
    Parameters:
    - sku_df: DataFrame with columns [date, demand]
    - policy_fn: function deciding reorder quantity
    - initial_inventory: starting stock level
    - lead_time_days: supplier lead time
    - holding_cost_per_unit: daily holding cost per unit
    - stockout_cost_per_unit: penalty per unit of unmet demand
    
    Returns:
    - Dictionary of simulation metrics
    """
    
    inventory = initial_inventory
    pipeline = []  # (arrival_day_index, quantity)
    
    holding_cost = 0
    stockout_cost = 0
    total_stockouts = 0
    
    for day_idx in range(len(sku_df)):
        demand = sku_df.loc[day_idx, "demand"]
        
        # Receive incoming orders
        arrivals = [qty for d, qty in pipeline if d == day_idx]
        inventory += sum(arrivals)
        pipeline = [(d, q) for d, q in pipeline if d > day_idx]
        
        # Fulfill demand
        if inventory >= demand:
            inventory -= demand
        else:
            unmet = demand - inventory
            total_stockouts += unmet
            stockout_cost += unmet * stockout_cost_per_unit
            inventory = 0
        
        # Holding cost for remaining inventory
        holding_cost += inventory * holding_cost_per_unit
        
        # Policy decision
        order_qty = policy_fn(inventory)
        if order_qty > 0:
            pipeline.append((day_idx + lead_time_days, order_qty))
    
    return {
        "holding_cost": holding_cost,
        "stockout_cost": stockout_cost,
        "total_cost": holding_cost + stockout_cost,
        "stockouts": total_stockouts
    }
