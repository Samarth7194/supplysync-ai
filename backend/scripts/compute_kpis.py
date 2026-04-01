"""
Compute real KPIs by running simulations on actual data.
Saves results to backend/data/cached_kpis.json.

Usage:
    cd backend
    python scripts/compute_kpis.py
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from ingestion.load_retail_data import load_sku_demand, get_top_skus
from simulation.enhanced_simulator import EnhancedInventorySimulator
from services.intelligent_inventory_service import IntelligentInventoryService
from uncertainty.kpi_utils import compute_kpis


def compute():
    print("=" * 60)
    print("SupplySync AI - KPI Computation")
    print("=" * 60)

    # Load daily demand data
    parquet_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "daily_demand.parquet")
    daily_df = pd.read_parquet(parquet_path)
    top_skus = get_top_skus(daily_df, min_days=60, top_n=10)
    print(f"Running simulation on {len(top_skus)} SKUs...")

    simulator = EnhancedInventorySimulator(holding_cost_per_unit=0.5, stockout_cost_per_unit=5.0)
    intelligent_service = IntelligentInventoryService(model=None)

    naive_total_cost = 0
    intelligent_total_cost = 0
    total_fill_rate = 0
    total_stockouts_naive = 0
    total_stockouts_intelligent = 0
    total_demand_all = 0
    holding_total = 0
    stockout_total = 0
    skus_simulated = 0

    for sku in top_skus:
        sku_df = load_sku_demand(sku)
        if len(sku_df) < 60:
            continue

        # Use last 90 days for simulation (or all if shorter)
        sim_df = sku_df.tail(90).reset_index(drop=True)
        total_demand = sim_df["demand"].sum()
        if total_demand == 0:
            continue

        try:
            # Run naive policy with demand-proportional params
            avg_demand = sim_df["demand"].mean()
            naive_threshold = int(avg_demand * 7)  # 1 week of demand
            naive_qty = int(avg_demand * 14)  # 2 weeks of demand

            naive_res = simulator.simulate_policy(
                sku_df=sim_df,
                policy_fn=simulator.naive_policy,
                policy_name="naive",
                lead_time_days=7,
                reorder_threshold=max(20, naive_threshold),
                reorder_qty=max(50, naive_qty),
            )

            # Run intelligent policy
            intel_res = simulator.simulate_policy(
                sku_df=sim_df,
                policy_fn=simulator.intelligent_policy,
                policy_name="intelligent",
                lead_time_days=7,
                intelligent_service=intelligent_service,
            )

            naive_total_cost += naive_res.total_cost
            intelligent_total_cost += intel_res.total_cost
            total_stockouts_naive += naive_res.stockouts
            total_stockouts_intelligent += intel_res.stockouts
            total_demand_all += total_demand
            holding_total += intel_res.holding_cost
            stockout_total += intel_res.stockout_cost
            total_fill_rate += intel_res.fill_rate
            skus_simulated += 1

            print(f"  {sku}: naive=${naive_res.total_cost:,.0f} vs intelligent=${intel_res.total_cost:,.0f} "
                  f"(fill_rate={intel_res.fill_rate:.2%})")
        except Exception as e:
            print(f"  {sku}: SKIPPED ({e})")
            continue

    if skus_simulated == 0:
        print("No SKUs could be simulated!")
        return

    # Aggregate KPIs
    avg_fill_rate = total_fill_rate / skus_simulated
    cost_savings_pct = ((naive_total_cost - intelligent_total_cost) / naive_total_cost * 100) if naive_total_cost > 0 else 0

    kpis = {
        "total_cost": round(intelligent_total_cost, 2),
        "fill_rate": round(avg_fill_rate, 4),
        "cost_savings_pct": round(cost_savings_pct, 1),
        "holding_cost": round(holding_total, 2),
        "stockout_cost": round(stockout_total, 2),
        "naive_total_cost": round(naive_total_cost, 2),
        "intelligent_total_cost": round(intelligent_total_cost, 2),
        "skus_analyzed": skus_simulated,
        "computed_at": datetime.now().isoformat(),
    }

    # Save to cache
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "cached_kpis.json")
    with open(cache_path, "w") as f:
        json.dump(kpis, f, indent=2)

    print(f"\nAggregated KPIs ({skus_simulated} SKUs):")
    print(f"  Total Cost (Intelligent): ${intelligent_total_cost:,.0f}")
    print(f"  Total Cost (Naive):       ${naive_total_cost:,.0f}")
    print(f"  Cost Savings:             {cost_savings_pct:.1f}%")
    print(f"  Avg Fill Rate:            {avg_fill_rate:.2%}")
    print(f"\nSaved to: {cache_path}")


if __name__ == "__main__":
    compute()
