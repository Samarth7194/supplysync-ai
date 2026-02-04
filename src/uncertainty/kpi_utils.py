# src/uncertainty/kpi_utils.py

def compute_kpis(simulation_results, total_demand):
    """
    Computes business KPIs from simulation results.

    This function converts raw simulation outputs into
    business-friendly metrics such as total cost and fill rate.

    Parameters:
    - simulation_results: dict returned by simulate_inventory()
    - total_demand: total demand over the simulation horizon

    Returns:
    - Dictionary containing KPI metrics
    """

    stockouts = simulation_results["stockouts"]

    # Fill rate = fulfilled demand / total demand
    fill_rate = 1 - (stockouts / total_demand if total_demand > 0 else 0)

    return {
        "holding_cost": simulation_results["holding_cost"],
        "stockout_cost": simulation_results["stockout_cost"],
        "total_cost": simulation_results["total_cost"],
        "fill_rate": round(fill_rate, 4),
    }
