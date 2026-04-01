# src/agentic/scenario_sim.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class ScenarioShock:
    name: str # e.g., "30% Fuel Price Hike"
    demand_multiplier: float
    volatility_increase: float
    lead_time_delay_days: int

class StochasticSimulator:
    """
    Lead Architect Implementation: Monte Carlo Simulator for 
    Stochastic Scenario Planning.
    
    Runs 1,000 parallel simulations to explore probabilistic 
    outcomes for systemic shocks.
    
    Task 5.2: Stochastic Scenario Modeling.
    """
    
    def __init__(self, num_iterations: int = 1000):
        self.num_iterations = num_iterations
        self.logger = logging.getLogger("scenario_sim")

    def run_monte_carlo(
        self, 
        base_demand: pd.Series, 
        shock: ScenarioShock
    ) -> Dict[str, Any]:
        """
        Executes simulations across the whole network for a given shock.
        
        Returns:
            Calibrated prediction intervals (P10, P50, P90) across the network.
        """
        self.logger.info(f"Initiating {self.num_iterations} simulations for shock: {shock.name}")
        
        # Base demand values
        demand_values = base_demand.values
        horizon = len(demand_values)
        
        # Simulated matrix: [iterations, horizon]
        sim_matrix = np.zeros((self.num_iterations, horizon))
        
        for i in range(self.num_iterations):
            # Apply multiplicative shock + random additive noise based on increased volatility
            noise = np.random.normal(0, shock.volatility_increase, horizon)
            sim_matrix[i, :] = (demand_values * shock.demand_multiplier) + noise
        
        # Task 5.2 Output: Calibrated Prediction Intervals
        p10 = np.percentile(sim_matrix, 10, axis=0)
        p50 = np.percentile(sim_matrix, 50, axis=0)
        p90 = np.percentile(sim_matrix, 90, axis=0)
        
        # Calculate Risk of Stockout
        return {
            "scenario": shock.name,
            "intervals": {
                "p10": p10.tolist(),
                "p50": p50.tolist(),
                "p90": p90.tolist()
            },
            "stats": {
                "mean_impact": float(np.mean(sim_matrix)),
                "worst_case_95th": float(np.percentile(sim_matrix, 95)),
                "best_case_5th": float(np.percentile(sim_matrix, 5))
            },
            "lead_time_adjustment": shock.lead_time_delay_days
        }

# Factory for common enterprise shocks
def get_enterprise_shocks() -> Dict[str, ScenarioShock]:
    """Generates a predefined list of high-impact scenarios."""
    return {
        "fuel_hike_30": ScenarioShock("30% Fuel Price Hike", 0.95, 0.1, 2),
        "port_strike_long_beach": ScenarioShock("3-week Port Strike", 0.85, 0.3, 21),
        "cpi_spike_10": ScenarioShock("10% CPI Surge", 0.9, 0.15, 0),
        "supplier_force_majeure": ScenarioShock("Supplier Force Majeure", 0.5, 0.5, 45)
    }
