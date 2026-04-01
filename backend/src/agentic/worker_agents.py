import sys
import os
from typing import Dict, List, Optional, Any
import pandas as pd

# Ensure the 'src' directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.intelligent_inventory_service import IntelligentInventoryService
from services.model_service import get_model_service

class TriageAgent:
    """Detects SKUs at risk of stockouts or overstock situations."""
    
    def __init__(self, ripple_engine: Optional[Any] = None):
        self.inventory_service = IntelligentInventoryService()
        self.ripple_engine = ripple_engine
    
    def analyze_sku_risk(self, sku: str, stock_level: int, history: pd.Series) -> Dict:
        """Analyze if a SKU needs urgent attention."""
        decision = self.inventory_service.get_intelligent_reorder_decision(
            sku=sku,
            current_stock=stock_level,
            demand_history=history
        )
        # Add risk classification
        risk_level = "LOW"
        if decision.get("action") == "REORDER":
            risk_level = "HIGH" if stock_level < decision.get("reorder_point") / 2 else "MEDIUM"
        
        return {
            "sku": sku,
            "risk_level": risk_level,
            "reasoning": decision.get("explainability", {}).get("decision_logic", "Legacy Logic Applied"),
            "base_decision": decision
        }

    def detect_systemic_risk(self, failing_supplier_ids: List[int], graph_data: Dict) -> Dict:
        """
        Task 5.1: Dependency Ripple Engine.
        Uses the GNN engine to propagate supplier failure across SKUs.
        """
        if self.ripple_engine:
            return self.ripple_engine.identify_systemic_impact(
                failing_supplier_ids=failing_supplier_ids,
                node_features=graph_data['features'],
                edge_index=graph_data['edge_index']
            )
        return {"error": "RippleEngine not initialized."}

class EnrichmentAgent:
    """Enriches SKU data with exogenous context like weather or supplier delays."""
    
    def fetch_exogenous_data(self, sku: str, region: str) -> Dict:
        return {
            "supplier_reliability": 0.85,
            "region_weather_risk": "Low",
            "market_trend": "Increasing Demand"
        }

class RemediationAgent:
    """Suggests corrective actions based on Triage and Enrichment findings."""
    
    def execute_remediation(self, triage_data: Dict, enriched_data: Dict) -> Dict:
        base_decision = triage_data.get("base_decision", {})
        adjusted_qty = base_decision.get("order_quantity", 0)
        reason = "Base quantity based on forecast."
        
        if enriched_data.get("market_trend") == "Increasing Demand":
            adjusted_qty = int(adjusted_qty * 1.1)
            reason = "Increased by 10% due to positive market trend."
            
        return {
            "sku": triage_data.get("sku"),
            "action_type": "PURCHASE_ORDER" if adjusted_qty > 0 else "OBSERVE",
            "final_quantity": adjusted_qty,
            "remediation_plan": [
                f"Identify risk: {triage_data.get('risk_level')}",
                f"Apply enrichment context: {enriched_data}",
                f"Adjust logic: {reason}"
            ]
        }

    def optimize_network_reorder(
        self, 
        sku_demands: List[Dict], 
        constraints: Dict,
        scenarios: Optional[List[Dict]] = None
    ) -> Dict:
        from ortools.linear_solver import pywraplp
        num_scenarios = len(scenarios) if scenarios else 100
        master_solver = pywraplp.Solver.CreateSolver('SCIP')
        if not master_solver:
            return {"error": "OR-Tools solver not available."}
            
        x_vars = {sku['id']: master_solver.IntVar(0, 10000, f"x_{sku['id']}") for sku in sku_demands}
        theta = master_solver.NumVar(0, 10e7, "theta")

        master_solver.Minimize(
            sum(x_vars[s['id']] * s.get('unit_cost', 10) for s in sku_demands) + theta
        )
        
        iteration_converged = False
        for i in range(10):
            status = master_solver.Solve()
            if status != pywraplp.Solver.OPTIMAL:
                break
                
            recourse_costs = []
            for s in range(num_scenarios):
                sc_multiplier = scenarios[s].get("demand_multiplier", 1.0) if scenarios else 1.0
                sc_cost = sum(max(0, (sku['base_demand'] * sc_multiplier) - x_vars[sku['id']].solution_value()) * sku.get('emergency_cost', 50) for sku in sku_demands)
                recourse_costs.append(sc_cost)
            
            e_recourse = sum(recourse_costs) / num_scenarios
            
            if theta.solution_value() >= e_recourse - 0.01:
                iteration_converged = True
                break
            else:
                master_solver.Add(theta >= e_recourse)

        return {
            "status": "BENDERS_OPTIMIZED" if iteration_converged else "FEASIBLE",
            "total_expected_cost": master_solver.Objective().Value(),
            "orders": {sku['id']: int(x_vars[sku['id']].solution_value()) for sku in sku_demands},
            "num_scenarios_evaluated": num_scenarios,
            "convergence_reached": iteration_converged
        }
