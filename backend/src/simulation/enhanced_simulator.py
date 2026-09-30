# src/simulation/enhanced_simulator.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from inventory.reorder_point import compute_reorder_decision
from services.intelligent_inventory_service import IntelligentInventoryService

@dataclass
class SimulationResults:
    """Results from inventory simulation.

    ``inventory_timeseries``/``stockout_timeseries`` cover only the *measured*
    window (after warm-up). Neither the naive, moving-average, nor intelligent
    policies take cost parameters, so these two arrays are independent of
    ``holding_cost_per_unit``/``stockout_cost_per_unit`` and can be re-weighted
    under a different cost ratio without re-running the simulation.
    """
    policy_name: str
    holding_cost: float
    stockout_cost: float
    total_cost: float
    stockouts: int
    fill_rate: float
    avg_inventory_level: float
    service_level: float
    inventory_timeseries: List[float]
    stockout_timeseries: List[float]
    reorder_events: List[Tuple[int, int]]  # (day_index, quantity)
    warmup_days: int
    measured_days: int

class EnhancedInventorySimulator:
    """
    Advanced simulator that compares reorder policies:
    1. Naive policy (fixed threshold)
    2. Moving-average reorder point with fixed safety stock
    3. ML forecast only
    4. ML + adaptive + uncertainty (full intelligent system)

    All policies decide off *inventory position* (on-hand + on-order), not
    on-hand alone, so a policy never re-orders on top of stock that is
    already inbound.
    """

    def __init__(self, holding_cost_per_unit: float = 1.0, stockout_cost_per_unit: float = 5.0):
        self.holding_cost_per_unit = holding_cost_per_unit
        self.stockout_cost_per_unit = stockout_cost_per_unit

    def naive_policy(self, inventory_position: float, reorder_threshold: int = 20, reorder_qty: int = 50) -> int:
        """Simple fixed threshold reorder policy."""
        if inventory_position < reorder_threshold:
            return reorder_qty
        return 0

    def moving_average_policy(
        self,
        inventory_position: float,
        recent_demand: pd.Series,
        lead_time_days: int,
        safety_stock: float,
        review_period_days: int = 7,
    ) -> int:
        """(s, S) reorder-point policy: fixed safety stock, moving-average demand rate.

        ``safety_stock`` is computed once, upfront, from the warm-up window and
        held fixed for the whole run -- unlike the intelligent policy's
        dynamic safety stock, which is recomputed from rolling forecast error.
        """
        avg_demand = float(recent_demand.mean()) if len(recent_demand) > 0 else 0.0
        reorder_point = avg_demand * lead_time_days + safety_stock
        if inventory_position < reorder_point:
            order_up_to = avg_demand * (lead_time_days + review_period_days) + safety_stock
            return max(0, int(round(order_up_to - inventory_position)))
        return 0

    def ml_forecast_policy(
        self,
        inventory_position: float,
        forecast: List[float],
        lead_time_days: int,
        sigma: float
    ) -> int:
        """ML forecast policy without adaptive features."""

        decision = compute_reorder_decision(
            sku="simulation",
            current_stock=inventory_position,
            forecast=forecast,
            sigma=sigma,
            lead_time_days=lead_time_days
        )

        return decision["order_quantity"]

    def intelligent_policy(
        self,
        sku: str,
        inventory_position: float,
        demand_history: pd.Series,
        lead_time_days: int,
        intelligent_service: IntelligentInventoryService,
        routing_service=None,
        uncertainty_service=None,
    ) -> int:
        """Full intelligent policy with adaptive forecasting and uncertainty."""

        decision = intelligent_service.get_intelligent_reorder_decision(
            sku=sku,
            current_stock=inventory_position,
            demand_history=demand_history,
            lead_time_days=lead_time_days,
            routing_service=routing_service,
            uncertainty_service=uncertainty_service,
        )

        return decision["order_quantity"]

    def simulate_policy(
        self,
        sku_df: pd.DataFrame,
        policy_name: str,
        lead_time_days: int = 7,
        initial_inventory: Optional[int] = None,
        warmup_days: int = 0,
        **policy_kwargs
    ) -> SimulationResults:
        """
        Simulate a single policy over time.

        Parameters:
        - sku_df: DataFrame with columns [date, demand]
        - policy_name: Which policy to run ("naive", "moving_average_rop",
          "ml_forecast", or "intelligent")
        - lead_time_days: Supplier lead time
        - initial_inventory: Starting inventory (auto-calculated if None)
        - warmup_days: Number of leading days simulated (so the initial
          inventory guess and empty pipeline can settle) but excluded from
          every reported metric.
        - policy_kwargs: Additional arguments for the policy function

        Returns:
        - SimulationResults object
        """

        if warmup_days < 0 or warmup_days >= len(sku_df):
            raise ValueError("warmup_days must be >= 0 and shorter than sku_df")

        if initial_inventory is None:
            # Start with 2 weeks of average demand
            initial_inventory = int(sku_df["demand"].mean() * 14)

        inventory = initial_inventory
        pipeline: list[tuple[int, int]] = []  # (arrival_day_index, quantity)

        holding_cost = 0.0
        stockout_cost = 0.0
        total_stockouts = 0.0
        total_demand = 0.0
        fulfilled_demand = 0.0

        inventory_timeseries: list[float] = []
        stockout_timeseries: list[float] = []
        reorder_events: list[tuple[int, int]] = []

        for day_idx in range(len(sku_df)):
            demand = sku_df.loc[day_idx, "demand"]
            measured = day_idx >= warmup_days

            # Receive incoming orders
            arrivals = [qty for d, qty in pipeline if d == day_idx]
            inventory += sum(arrivals)

            # Remove arrived orders from pipeline
            pipeline = [(d, qty) for d, qty in pipeline if d > day_idx]

            # Fulfill demand from on-hand stock
            if inventory >= demand:
                inventory -= demand
                unmet = 0.0
                fulfilled = demand
            else:
                unmet = demand - inventory
                fulfilled = inventory
                inventory = 0

            if measured:
                total_demand += demand
                fulfilled_demand += fulfilled
                total_stockouts += unmet
                stockout_cost += unmet * self.stockout_cost_per_unit
                holding_cost += inventory * self.holding_cost_per_unit
                inventory_timeseries.append(inventory)
                stockout_timeseries.append(unmet)

            # Inventory *position* (on-hand + already on-order) is what every
            # policy decides on -- using on-hand alone would keep re-ordering
            # on top of stock that is already inbound.
            pipeline_qty = sum(qty for _, qty in pipeline)
            inventory_position = inventory + pipeline_qty

            # Policy decision
            if policy_name == "naive":
                order_qty = self.naive_policy(inventory_position, **policy_kwargs)
            elif policy_name == "moving_average_rop":
                window = policy_kwargs.get("review_window_days", 28)
                recent_demand = sku_df.loc[max(0, day_idx - window + 1):day_idx, "demand"]
                order_qty = self.moving_average_policy(
                    inventory_position,
                    recent_demand,
                    lead_time_days,
                    policy_kwargs.get("safety_stock", 0.0),
                    policy_kwargs.get("review_period_days", 7),
                )
            elif policy_name == "ml_forecast":
                order_qty = self.ml_forecast_policy(
                    inventory_position,
                    policy_kwargs.get("forecast", []),
                    lead_time_days,
                    policy_kwargs.get("sigma", 1.0)
                )
            elif policy_name == "candidate_forecast":
                # Re-derive the forecast fresh each day from demand observed
                # so far, like "intelligent" does, but for an arbitrary
                # forecast_fn(history, horizon) -> list[float] -- used to
                # compare candidate forecast methods by simulated cost
                # instead of WAPE (see scripts/evaluate_highly_intermittent_policy.py).
                current_demand_history = sku_df.loc[:day_idx, "demand"]
                forecast_fn = policy_kwargs["forecast_fn"]
                forecast = forecast_fn(current_demand_history, lead_time_days)
                sigma = float(current_demand_history.std()) if len(current_demand_history) > 1 else 0.0
                order_qty = self.ml_forecast_policy(inventory_position, forecast, lead_time_days, sigma)
            elif policy_name == "intelligent":
                # Get demand history up to current point
                current_demand_history = sku_df.loc[:day_idx, "demand"]
                order_qty = self.intelligent_policy(
                    "simulation_sku",
                    inventory_position,
                    current_demand_history,
                    lead_time_days,
                    policy_kwargs.get("intelligent_service"),
                    routing_service=policy_kwargs.get("routing_service"),
                    uncertainty_service=policy_kwargs.get("uncertainty_service"),
                )
            else:
                order_qty = 0

            if order_qty > 0:
                pipeline.append((day_idx + lead_time_days, order_qty))
                if measured:
                    reorder_events.append((day_idx, order_qty))

        # Calculate metrics (measured window only)
        total_cost = holding_cost + stockout_cost
        fill_rate = fulfilled_demand / total_demand if total_demand > 0 else 0
        avg_inventory_level = float(np.mean(inventory_timeseries)) if inventory_timeseries else 0.0
        service_level = 1 - (total_stockouts / total_demand) if total_demand > 0 else 0

        return SimulationResults(
            policy_name=policy_name,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            total_cost=total_cost,
            stockouts=int(total_stockouts),
            fill_rate=fill_rate,
            avg_inventory_level=avg_inventory_level,
            service_level=service_level,
            inventory_timeseries=inventory_timeseries,
            stockout_timeseries=stockout_timeseries,
            reorder_events=reorder_events,
            warmup_days=warmup_days,
            measured_days=len(sku_df) - warmup_days,
        )


def reweight_cost(
    result: SimulationResults,
    holding_cost_per_unit: float,
    stockout_cost_per_unit: float,
) -> tuple[float, float, float]:
    """Recompute (holding_cost, stockout_cost, total_cost) under a different
    cost ratio from an already-recorded trajectory, without re-simulating.

    Valid because none of the policy functions take cost parameters, so the
    inventory/stockout trajectory a policy produces does not depend on
    ``holding_cost_per_unit``/``stockout_cost_per_unit``.
    """
    holding = sum(result.inventory_timeseries) * holding_cost_per_unit
    stockout = sum(result.stockout_timeseries) * stockout_cost_per_unit
    return holding, stockout, holding + stockout
