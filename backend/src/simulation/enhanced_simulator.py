# src/simulation/enhanced_simulator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from inventory.reorder_point import compute_reorder_decision
from services.adaptive_forecasting_service import adaptive_forecast, classify_sku_demand_pattern
from services.intelligent_inventory_service import IntelligentInventoryService

@dataclass
class SimulationResults:
    """Results from inventory simulation."""
    policy_name: str
    holding_cost: float
    stockout_cost: float
    total_cost: float
    stockouts: int
    fill_rate: float
    avg_inventory_level: float
    service_level: float
    inventory_timeseries: List[float]
    reorder_events: List[Tuple[int, int]]  # (day_index, quantity)

class EnhancedInventorySimulator:
    """
    Advanced simulator that compares 3 policies:
    1. Naive policy (fixed threshold)
    2. ML forecast only
    3. ML + adaptive + uncertainty (full intelligent system)
    """
    
    def __init__(self, holding_cost_per_unit: float = 1.0, stockout_cost_per_unit: float = 5.0):
        self.holding_cost_per_unit = holding_cost_per_unit
        self.stockout_cost_per_unit = stockout_cost_per_unit
        
    def naive_policy(self, current_inventory: int, reorder_threshold: int = 20, reorder_qty: int = 50) -> int:
        """Simple fixed threshold reorder policy."""
        if current_inventory < reorder_threshold:
            return reorder_qty
        return 0
    
    def ml_forecast_policy(
        self, 
        current_inventory: int,
        forecast: List[float],
        lead_time_days: int,
        sigma: float
    ) -> int:
        """ML forecast policy without adaptive features."""
        
        decision = compute_reorder_decision(
            sku="simulation",
            current_stock=current_inventory,
            forecast=forecast,
            sigma=sigma,
            lead_time_days=lead_time_days
        )
        
        return decision["order_quantity"]
    
    def intelligent_policy(
        self,
        sku: str,
        current_inventory: int,
        demand_history: pd.Series,
        lead_time_days: int,
        intelligent_service: IntelligentInventoryService
    ) -> int:
        """Full intelligent policy with adaptive forecasting and uncertainty."""
        
        decision = intelligent_service.get_intelligent_reorder_decision(
            sku=sku,
            current_stock=current_inventory,
            demand_history=demand_history,
            lead_time_days=lead_time_days
        )
        
        return decision["order_quantity"]
    
    def simulate_policy(
        self,
        sku_df: pd.DataFrame,
        policy_fn,
        policy_name: str,
        lead_time_days: int = 7,
        initial_inventory: Optional[int] = None,
        **policy_kwargs
    ) -> SimulationResults:
        """
        Simulate a single policy over time.
        
        Parameters:
        - sku_df: DataFrame with columns [date, demand]
        - policy_fn: Function that decides reorder quantity
        - policy_name: Name of the policy for results
        - lead_time_days: Supplier lead time
        - initial_inventory: Starting inventory (auto-calculated if None)
        - policy_kwargs: Additional arguments for policy function
        
        Returns:
        - SimulationResults object
        """
        
        if initial_inventory is None:
            # Start with 2 weeks of average demand
            initial_inventory = int(sku_df["demand"].mean() * 14)
        
        inventory = initial_inventory
        pipeline = []  # (arrival_day_index, quantity)
        
        holding_cost = 0
        stockout_cost = 0
        total_stockouts = 0
        total_demand = 0
        fulfilled_demand = 0
        
        inventory_timeseries = []
        reorder_events = []
        
        for day_idx in range(len(sku_df)):
            demand = sku_df.loc[day_idx, "demand"]
            total_demand += demand
            
            # Receive incoming orders
            arrivals = [qty for d, qty in pipeline if d == day_idx]
            inventory += sum(arrivals)
            
            # Remove arrived orders from pipeline
            pipeline = [(d, qty) for d, qty in pipeline if d > day_idx]
            
            # Fulfill demand
            if inventory >= demand:
                inventory -= demand
                fulfilled_demand += demand
            else:
                unmet = demand - inventory
                total_stockouts += unmet
                stockout_cost += unmet * self.stockout_cost_per_unit
                fulfilled_demand += inventory
                inventory = 0
            
            # Holding cost for remaining inventory
            holding_cost += inventory * self.holding_cost_per_unit
            
            # Record inventory level
            inventory_timeseries.append(inventory)
            
            # Policy decision
            if policy_name == "naive":
                order_qty = self.naive_policy(inventory, **policy_kwargs)
            elif policy_name == "ml_forecast":
                order_qty = self.ml_forecast_policy(
                    inventory, 
                    policy_kwargs.get("forecast", []),
                    lead_time_days,
                    policy_kwargs.get("sigma", 1.0)
                )
            elif policy_name == "intelligent":
                # Get demand history up to current point
                current_demand_history = sku_df.loc[:day_idx, "demand"]
                order_qty = self.intelligent_policy(
                    "simulation_sku",
                    inventory,
                    current_demand_history,
                    lead_time_days,
                    policy_kwargs.get("intelligent_service")
                )
            else:
                order_qty = 0
            
            if order_qty > 0:
                pipeline.append((day_idx + lead_time_days, order_qty))
                reorder_events.append((day_idx, order_qty))
        
        # Calculate metrics
        total_cost = holding_cost + stockout_cost
        fill_rate = fulfilled_demand / total_demand if total_demand > 0 else 0
        avg_inventory_level = np.mean(inventory_timeseries)
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
            reorder_events=reorder_events
        )
    
    def compare_policies(
        self,
        sku_df: pd.DataFrame,
        intelligent_service: IntelligentInventoryService,
        lead_time_days: int = 7,
        forecast_horizon: int = 7
    ) -> Dict[str, SimulationResults]:
        """
        Compare all three policies on the same data.
        
        Parameters:
        - sku_df: Demand data for SKU
        - intelligent_service: Trained intelligent inventory service
        - lead_time_days: Supplier lead time
        - forecast_horizon: Forecast horizon for ML policy
        
        Returns:
        - Dictionary of results by policy name
        """
        
        results = {}
        
        # 1. Naive Policy
        results["naive"] = self.simulate_policy(
            sku_df=sku_df,
            policy_fn=self.naive_policy,
            policy_name="naive",
            lead_time_days=lead_time_days,
            reorder_threshold=20,
            reorder_qty=50
        )
        
        # 2. ML Forecast Policy
        # Generate simple forecasts for demonstration
        demand_series = sku_df["demand"]
        sigma = demand_series.std()
        
        # Simple moving average forecast
        forecast = [demand_series.tail(7).mean()] * forecast_horizon
        
        results["ml_forecast"] = self.simulate_policy(
            sku_df=sku_df,
            policy_fn=self.ml_forecast_policy,
            policy_name="ml_forecast",
            lead_time_days=lead_time_days,
            forecast=forecast,
            sigma=sigma
        )
        
        # 3. Intelligent Policy
        results["intelligent"] = self.simulate_policy(
            sku_df=sku_df,
            policy_fn=self.intelligent_policy,
            policy_name="intelligent",
            lead_time_days=lead_time_days,
            intelligent_service=intelligent_service
        )
        
        return results
    
    def generate_performance_report(self, results: Dict[str, SimulationResults]) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Parameters:
        - results: Simulation results by policy
        
        Returns:
        - Dictionary with performance insights
        """
        
        # Calculate improvements
        naive_cost = results["naive"].total_cost
        ml_cost = results["ml_forecast"].total_cost
        intelligent_cost = results["intelligent"].total_cost
        
        ml_improvement = ((naive_cost - ml_cost) / naive_cost) * 100
        intelligent_improvement = ((naive_cost - intelligent_cost) / naive_cost) * 100
        ml_vs_intelligent = ((ml_cost - intelligent_cost) / ml_cost) * 100
        
        return {
            "cost_analysis": {
                "naive_cost": naive_cost,
                "ml_forecast_cost": ml_cost,
                "intelligent_cost": intelligent_cost,
                "ml_vs_naive_improvement_pct": round(ml_improvement, 2),
                "intelligent_vs_naive_improvement_pct": round(intelligent_improvement, 2),
                "intelligent_vs_ml_improvement_pct": round(ml_vs_intelligent, 2)
            },
            "service_analysis": {
                "best_service_level": max(results.keys(), key=lambda k: results[k].service_level),
                "best_fill_rate": max(results.keys(), key=lambda k: results[k].fill_rate),
                "lowest_stockouts": min(results.keys(), key=lambda k: results[k].stockouts)
            },
            "efficiency_analysis": {
                "lowest_avg_inventory": min(results.keys(), key=lambda k: results[k].avg_inventory_level),
                "inventory_efficiency": {
                    policy: round(results[policy].total_cost / results[policy].avg_inventory_level, 2)
                    for policy in results.keys()
                }
            },
            "recommendation": {
                "best_overall": min(results.keys(), key=lambda k: results[k].total_cost),
                "best_for_service": max(results.keys(), key=lambda k: results[k].service_level),
                "most_efficient": min(results.keys(), key=lambda k: results[k].avg_inventory_level)
            }
        }
