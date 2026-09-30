"""
Step 2 (forecast-validity task): highly-intermittent demand must not be
selected by WAPE -- predict-zero always wins WAPE there (most days are zero)
but never orders anything, so a WAPE table can't tell us whether
"conservative x1.5" is actually the right policy.

This compares candidate forecast sources for highly-intermittent SKUs by
simulated inventory cost instead: each candidate is re-derived fresh every
day from demand observed so far (enhanced_simulator's "candidate_forecast"
policy) and run through the real reorder-decision math, on the VALIDATION
window only (never the held-out window).

Usage:
    cd backend
    python scripts/evaluate_highly_intermittent_policy.py
"""

import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from evaluation import backtest as bt
from services.adaptive_forecasting_service import (
    classify_sku_demand_pattern,
    conservative_forecast,
    croston_forecast,
    simple_average_forecast,
)
from simulation.enhanced_simulator import EnhancedInventorySimulator, reweight_cost

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
WARMUP_DAYS = 14
MEASURED_DAYS = 90
LEAD_TIME_DAYS = 7
HOLDING_COST_PER_UNIT = 0.5
STOCKOUT_COST_PER_UNIT = 5.0
# Same values Step 7's KPI simulation checks -- a policy choice that only
# wins at one arbitrary ratio isn't a robust choice.
SENSITIVITY_RATIOS = [2.0, 5.0, 10.0, 20.0]

CANDIDATES = {
    "conservative_x1.0": partial(conservative_forecast, buffer=1.0),
    "conservative_x1.5_old_default": partial(conservative_forecast, buffer=1.5),
    "conservative_x2.0": partial(conservative_forecast, buffer=2.0),
    "croston_sba": croston_forecast,
    "simple_average_7d": simple_average_forecast,
}


def main():
    parquet_path = os.path.join(BACKEND_DIR, "..", "data", "processed", "daily_demand.parquet")
    daily = pd.read_parquet(parquet_path)
    daily["date"] = pd.to_datetime(daily["date"])
    dataset_end = daily["date"].max()
    validation_end = bt.validation_dataset_end(dataset_end)
    daily = daily[daily["date"] <= validation_end]

    skus = bt.select_eval_skus(daily, cutoff=validation_end, min_active_days=60, max_skus=None)
    series_by_sku = bt.prepare_sku_series(daily, skus=skus, pad_to=validation_end)

    highly_intermittent = {
        sku: s for sku, s in series_by_sku.items()
        if classify_sku_demand_pattern(s.tail(60)) == "highly_intermittent"
    }
    print(f"Validation window ends {validation_end.date()}; "
          f"{len(highly_intermittent)} highly-intermittent SKUs (of {len(series_by_sku)} evaluated).")

    simulator = EnhancedInventorySimulator(
        holding_cost_per_unit=HOLDING_COST_PER_UNIT, stockout_cost_per_unit=STOCKOUT_COST_PER_UNIT
    )
    totals = {name: {"cost": 0.0, "demand": 0.0, "fulfilled": 0.0, "skus": 0} for name in CANDIDATES}
    results_by_candidate = {name: [] for name in CANDIDATES}

    for sku, series in highly_intermittent.items():
        if len(series) < WARMUP_DAYS + MEASURED_DAYS:
            continue
        sim_df = series.tail(WARMUP_DAYS + MEASURED_DAYS).reset_index(drop=True).rename("demand").to_frame()
        measured_demand = sim_df["demand"].iloc[WARMUP_DAYS:].sum()
        if measured_demand == 0:
            continue
        for name, fn in CANDIDATES.items():
            result = simulator.simulate_policy(
                sku_df=sim_df, policy_name="candidate_forecast", lead_time_days=LEAD_TIME_DAYS,
                warmup_days=WARMUP_DAYS, forecast_fn=fn,
            )
            totals[name]["cost"] += result.total_cost
            totals[name]["demand"] += measured_demand
            totals[name]["fulfilled"] += result.fill_rate * measured_demand
            totals[name]["skus"] += 1
            results_by_candidate[name].append(result)

    print(f"\n{'candidate':<28}{'total_cost':>14}{'fill_rate':>12}{'skus':>7}")
    for name, t in sorted(totals.items(), key=lambda kv: kv[1]["cost"]):
        fill_rate = t["fulfilled"] / t["demand"] if t["demand"] else 0.0
        print(f"{name:<28}{t['cost']:>14,.0f}{fill_rate:>12.2%}{t['skus']:>7}")

    print(f"\nSensitivity to the stockout:holding cost ratio (re-weighting recorded trajectories, no re-simulation):")
    print(f"{'ratio':>8}" + "".join(f"{name:>28}" for name in CANDIDATES))
    sensitivity = {}
    for ratio in SENSITIVITY_RATIOS:
        holding_unit = HOLDING_COST_PER_UNIT
        stockout_unit = holding_unit * ratio
        row = f"{ratio:>7.0f}:1"
        sensitivity[ratio] = {}
        for name in CANDIDATES:
            cost = sum(reweight_cost(r, holding_unit, stockout_unit)[2] for r in results_by_candidate[name])
            sensitivity[ratio][name] = round(cost, 2)
            row += f"{cost:>28,.0f}"
        print(row)

    winner = min(totals, key=lambda name: totals[name]["cost"])
    out_path = os.path.join(BACKEND_DIR, "data", "highly_intermittent_policy_evaluation.json")
    with open(out_path, "w") as f:
        json.dump({
            "validation_window_end": str(validation_end.date()),
            "n_highly_intermittent_skus": len(highly_intermittent),
            "n_skus_simulated": next(iter(totals.values()))["skus"],
            "warmup_days": WARMUP_DAYS,
            "measured_days": MEASURED_DAYS,
            "cost_assumptions": {"holding_cost_per_unit": HOLDING_COST_PER_UNIT, "stockout_cost_per_unit": STOCKOUT_COST_PER_UNIT},
            "candidates": {
                name: {
                    "total_cost": round(t["cost"], 2),
                    "fill_rate": round(t["fulfilled"] / t["demand"], 4) if t["demand"] else None,
                }
                for name, t in totals.items()
            },
            "sensitivity_to_cost_ratio": sensitivity,
            "winner_by_total_cost": winner,
            "winner_at_every_tested_ratio": all(
                min(sensitivity[r], key=sensitivity[r].get) == winner for r in SENSITIVITY_RATIOS
            ),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
