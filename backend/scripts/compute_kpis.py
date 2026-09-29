"""
Compute real KPIs by running simulations on actual data.
Saves results to backend/data/cached_kpis.json.

What this actually measures
----------------------------
For each of ``TOP_N_SKUS`` SKUs (by total volume, from the full cleaned
dataset), the last ``WARMUP_DAYS + MEASURED_DAYS`` days of demand are
simulated. The first ``WARMUP_DAYS`` let the initial inventory guess and
empty order pipeline settle; only the remaining ``MEASURED_DAYS`` count
toward every reported cost/fill-rate/stockout number. Three policies are
compared on the *same* demand trajectory per SKU:

  - naive: fixed reorder threshold/quantity (scaled to the SKU's demand rate)
  - moving_average_rop: (s, S) reorder-point policy with a fixed safety stock
    (computed once from the warm-up window) and a rolling demand-rate estimate
  - intelligent: the real production forecast path (loaded LightGBM model,
    evidence-based routing, dynamic safety stock, business constraints) --
    not ``model=None``

The headline "cost savings" is measured against whichever baseline (naive or
moving_average_rop) actually has the lower total cost for that run, not
naive unconditionally, and a 95% CI across SKUs is reported alongside the
mean. Sensitivity to the stockout/holding cost ratio is computed by
re-weighting each SKU's already-recorded inventory/stockout trajectory under
several ratios (valid because no policy here takes cost parameters), not by
re-running the simulation.

Usage:
    cd backend
    python scripts/compute_kpis.py
"""

import sys
import os
import json
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from ingestion.load_retail_data import load_sku_demand, get_top_skus
from simulation.enhanced_simulator import EnhancedInventorySimulator, reweight_cost
from services.intelligent_inventory_service import IntelligentInventoryService
from services.model_service import ModelService, ModelArtifactValidationError
from services.model_routing_service import ModelRoutingService
from config.forecasting import load_forecasting_settings

MODEL_NAME = "lightgbm_demand_forecast"
TOP_N_SKUS = 50
WARMUP_DAYS = 14
MEASURED_DAYS = 90
LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.645  # one-sided z-score for a 95% service level
HOLDING_COST_PER_UNIT = 0.5
STOCKOUT_COST_PER_UNIT = 5.0
# Alternate stockout:holding cost ratios used only to re-weight recorded
# trajectories after simulation -- the ratio never changes how a policy
# decides to reorder.
SENSITIVITY_RATIOS = [2.0, 5.0, 10.0, 20.0]

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")


def _load_production_model():
    """Load the same artifact main.py loads at startup (file-system path;
    this script has no DB session, matching main.py's no-DB-active-artifact
    fallback in runtime_model_service.load_runtime_model)."""
    model_dir = os.path.join(BACKEND_DIR, "saved_models")
    service = ModelService(model_dir=model_dir)
    try:
        model = service.load_model(MODEL_NAME)
        metadata = service.get_model_metadata(MODEL_NAME) or {}
        return model, metadata.get("features")
    except (FileNotFoundError, ModelArtifactValidationError) as e:
        print(f"WARNING: could not load production model ({e}); "
              f"the intelligent policy will fall back to simple_average forecasts.")
        return None, None


def _mean_ci95(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / math.sqrt(n)
    return mean, mean - half_width, mean + half_width


def compute():
    print("=" * 60)
    print("SupplySync AI - KPI Computation")
    print("=" * 60)

    parquet_path = os.path.join(BACKEND_DIR, "..", "data", "processed", "daily_demand.parquet")
    daily_df = pd.read_parquet(parquet_path)
    dataset_sku_count = daily_df["StockCode"].nunique()
    top_skus = get_top_skus(daily_df, min_days=WARMUP_DAYS + MEASURED_DAYS, top_n=TOP_N_SKUS)
    print(f"Dataset has {dataset_sku_count} unique SKUs; simulating the top {len(top_skus)} by volume "
          f"with >= {WARMUP_DAYS + MEASURED_DAYS} days of history.")

    model, feature_columns = _load_production_model()
    print(f"Production model loaded: {model is not None} "
          f"({len(feature_columns) if feature_columns else 0} features)")
    routing_service = ModelRoutingService(
        settings=load_forecasting_settings(),
        offline_evaluation_path=os.path.join(BACKEND_DIR, "data", "forecast_evaluation.json"),
    )

    simulator = EnhancedInventorySimulator(
        holding_cost_per_unit=HOLDING_COST_PER_UNIT,
        stockout_cost_per_unit=STOCKOUT_COST_PER_UNIT,
    )
    intelligent_service = IntelligentInventoryService(model=model, model_feature_columns=feature_columns)

    per_sku_results = []  # list of dicts: naive/mov_avg/intelligent SimulationResults + sku

    for sku in top_skus:
        sku_df = load_sku_demand(sku)
        if len(sku_df) < WARMUP_DAYS + MEASURED_DAYS:
            continue

        sim_df = sku_df.tail(WARMUP_DAYS + MEASURED_DAYS).reset_index(drop=True)
        warmup_demand = sim_df["demand"].iloc[:WARMUP_DAYS]
        measured_demand_total = sim_df["demand"].iloc[WARMUP_DAYS:].sum()
        if measured_demand_total == 0:
            continue

        try:
            avg_demand = float(warmup_demand.mean())
            naive_threshold = int(avg_demand * LEAD_TIME_DAYS)
            naive_qty = int(avg_demand * LEAD_TIME_DAYS * 2)

            naive_res = simulator.simulate_policy(
                sku_df=sim_df,
                policy_name="naive",
                lead_time_days=LEAD_TIME_DAYS,
                warmup_days=WARMUP_DAYS,
                reorder_threshold=max(20, naive_threshold),
                reorder_qty=max(50, naive_qty),
            )

            sigma = float(warmup_demand.std())
            safety_stock = SERVICE_LEVEL_Z * sigma * math.sqrt(LEAD_TIME_DAYS) if pd.notna(sigma) else 0.0

            mov_avg_res = simulator.simulate_policy(
                sku_df=sim_df,
                policy_name="moving_average_rop",
                lead_time_days=LEAD_TIME_DAYS,
                warmup_days=WARMUP_DAYS,
                safety_stock=safety_stock,
            )

            intel_res = simulator.simulate_policy(
                sku_df=sim_df,
                policy_name="intelligent",
                lead_time_days=LEAD_TIME_DAYS,
                warmup_days=WARMUP_DAYS,
                intelligent_service=intelligent_service,
                routing_service=routing_service,
            )

            per_sku_results.append({
                "sku": sku,
                "naive": naive_res,
                "moving_average_rop": mov_avg_res,
                "intelligent": intel_res,
            })

            strongest = "naive" if naive_res.total_cost <= mov_avg_res.total_cost else "moving_average_rop"
            print(f"  {sku}: naive=${naive_res.total_cost:,.0f} mov_avg_rop=${mov_avg_res.total_cost:,.0f} "
                  f"intelligent=${intel_res.total_cost:,.0f} (strongest baseline={strongest}, "
                  f"fill_rate={intel_res.fill_rate:.2%})")
        except Exception as e:
            print(f"  {sku}: SKIPPED ({type(e).__name__}: {e})")
            continue

    if not per_sku_results:
        print("No SKUs could be simulated!")
        return

    skus_simulated = len(per_sku_results)

    naive_total_cost = sum(r["naive"].total_cost for r in per_sku_results)
    mov_avg_total_cost = sum(r["moving_average_rop"].total_cost for r in per_sku_results)
    intelligent_total_cost = sum(r["intelligent"].total_cost for r in per_sku_results)
    holding_total = sum(r["intelligent"].holding_cost for r in per_sku_results)
    stockout_total = sum(r["intelligent"].stockout_cost for r in per_sku_results)
    avg_fill_rate = float(np.mean([r["intelligent"].fill_rate for r in per_sku_results]))

    strongest_baseline_method = "naive" if naive_total_cost <= mov_avg_total_cost else "moving_average_rop"
    strongest_baseline_total_cost = min(naive_total_cost, mov_avg_total_cost)

    # Headline cost-savings number: intelligent vs. whichever baseline is
    # actually stronger, computed per-SKU so a 95% CI can be reported too.
    per_sku_savings_vs_strongest = []
    per_sku_savings_vs_naive = []
    for r in per_sku_results:
        baseline_cost = r[strongest_baseline_method].total_cost
        if baseline_cost > 0:
            per_sku_savings_vs_strongest.append(
                (baseline_cost - r["intelligent"].total_cost) / baseline_cost * 100
            )
        if r["naive"].total_cost > 0:
            per_sku_savings_vs_naive.append(
                (r["naive"].total_cost - r["intelligent"].total_cost) / r["naive"].total_cost * 100
            )

    savings_mean, savings_ci_low, savings_ci_high = _mean_ci95(per_sku_savings_vs_strongest)
    cost_savings_pct = (
        (naive_total_cost - intelligent_total_cost) / naive_total_cost * 100 if naive_total_cost > 0 else 0
    )

    # Sensitivity to the stockout/holding cost ratio: re-weight the already
    # recorded trajectories, do not re-simulate.
    sensitivity = []
    for ratio in SENSITIVITY_RATIOS:
        holding_unit = HOLDING_COST_PER_UNIT
        stockout_unit = holding_unit * ratio
        naive_cost = mov_avg_cost = intel_cost = 0.0
        for r in per_sku_results:
            _, _, nc = reweight_cost(r["naive"], holding_unit, stockout_unit)
            _, _, mc = reweight_cost(r["moving_average_rop"], holding_unit, stockout_unit)
            _, _, ic = reweight_cost(r["intelligent"], holding_unit, stockout_unit)
            naive_cost += nc
            mov_avg_cost += mc
            intel_cost += ic
        strongest_cost = min(naive_cost, mov_avg_cost)
        savings_pct = (strongest_cost - intel_cost) / strongest_cost * 100 if strongest_cost > 0 else 0
        sensitivity.append({
            "stockout_to_holding_ratio": ratio,
            "naive_total_cost": round(naive_cost, 2),
            "moving_average_rop_total_cost": round(mov_avg_cost, 2),
            "intelligent_total_cost": round(intel_cost, 2),
            "cost_savings_vs_strongest_baseline_pct": round(savings_pct, 1),
        })

    kpis = {
        # Backward-compatible fields (unchanged meaning: intelligent vs naive).
        "total_cost": round(intelligent_total_cost, 2),
        "fill_rate": round(avg_fill_rate, 4),
        "cost_savings_pct": round(cost_savings_pct, 1),
        "holding_cost": round(holding_total, 2),
        "stockout_cost": round(stockout_total, 2),
        "naive_total_cost": round(naive_total_cost, 2),
        "intelligent_total_cost": round(intelligent_total_cost, 2),
        "skus_analyzed": skus_simulated,
        "computed_at": datetime.now().isoformat(),

        # New, more honest fields.
        "moving_average_rop_total_cost": round(mov_avg_total_cost, 2),
        "strongest_baseline_method": strongest_baseline_method,
        "strongest_baseline_total_cost": round(strongest_baseline_total_cost, 2),
        "cost_savings_vs_strongest_baseline_pct": round(savings_mean, 1),
        "cost_savings_vs_strongest_baseline_ci95": [round(savings_ci_low, 1), round(savings_ci_high, 1)],
        "cost_savings_vs_naive_pct": round(float(np.mean(per_sku_savings_vs_naive)) if per_sku_savings_vs_naive else 0.0, 1),
        "model_loaded": model is not None,
        "dataset_sku_count": int(dataset_sku_count),
        "skus_requested": TOP_N_SKUS,
        "warmup_days": WARMUP_DAYS,
        "measured_days": MEASURED_DAYS,
        "lead_time_days": LEAD_TIME_DAYS,
        "policies_compared": ["naive_fixed_threshold", "moving_average_reorder_point", "intelligent"],
        "sensitivity_to_cost_ratio": sensitivity,
    }

    cache_dir = os.path.join(BACKEND_DIR, "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "cached_kpis.json")
    with open(cache_path, "w") as f:
        json.dump(kpis, f, indent=2)

    print(f"\nAggregated KPIs ({skus_simulated} SKUs, {WARMUP_DAYS}-day warm-up + {MEASURED_DAYS} measured days):")
    print(f"  Total Cost (Intelligent):        ${intelligent_total_cost:,.0f}")
    print(f"  Total Cost (Naive):               ${naive_total_cost:,.0f}")
    print(f"  Total Cost (Moving-avg ROP):       ${mov_avg_total_cost:,.0f}")
    print(f"  Strongest baseline:               {strongest_baseline_method} (${strongest_baseline_total_cost:,.0f})")
    print(f"  Cost savings vs strongest baseline: {savings_mean:.1f}% (95% CI [{savings_ci_low:.1f}, {savings_ci_high:.1f}])")
    print(f"  Avg Fill Rate:                     {avg_fill_rate:.2%}")
    print(f"\nSaved to: {cache_path}")


if __name__ == "__main__":
    compute()
