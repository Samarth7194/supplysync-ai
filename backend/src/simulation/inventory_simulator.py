# src/simulation/inventory_simulator.py

def simulate_inventory(
    sku_df,
    policy_fn,
    initial_inventory,
    lead_time_days,
    holding_cost_per_unit,
    stockout_cost_per_unit,
):
    """Simulates day-by-day inventory operations for a single SKU.

    Parameters
    ----------
    sku_df : DataFrame with column ``demand`` (row order = time)
    policy_fn : callable(inventory) -> order_qty
    initial_inventory : int
    lead_time_days : int
    holding_cost_per_unit : float (per unit per day)
    stockout_cost_per_unit : float (per unit of unmet demand)

    Returns dict with: holding_cost, stockout_cost, total_cost,
    stockouts (units unmet), fill_rate, service_level.
    """
    inventory = initial_inventory
    pipeline = []  # (arrival_day_index, quantity)

    holding_cost = 0.0
    stockout_cost = 0.0
    total_stockouts = 0
    total_demand = 0
    fulfilled_demand = 0

    for day_idx in range(len(sku_df)):
        demand = sku_df.loc[day_idx, "demand"]
        total_demand += demand

        arrivals = [qty for d, qty in pipeline if d == day_idx]
        inventory += sum(arrivals)
        pipeline = [(d, q) for d, q in pipeline if d > day_idx]

        if inventory >= demand:
            inventory -= demand
            fulfilled_demand += demand
        else:
            unmet = demand - inventory
            total_stockouts += unmet
            stockout_cost += unmet * stockout_cost_per_unit
            fulfilled_demand += inventory
            inventory = 0

        holding_cost += inventory * holding_cost_per_unit

        order_qty = policy_fn(inventory)
        if order_qty > 0:
            pipeline.append((day_idx + lead_time_days, order_qty))

    fill_rate = (fulfilled_demand / total_demand) if total_demand > 0 else 1.0
    service_level = 1 - (total_stockouts / total_demand) if total_demand > 0 else 1.0

    return {
        "holding_cost": holding_cost,
        "stockout_cost": stockout_cost,
        "total_cost": holding_cost + stockout_cost,
        "stockouts": total_stockouts,
        "fill_rate": fill_rate,
        "service_level": service_level,
    }
