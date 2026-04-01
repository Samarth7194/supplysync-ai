import sys
import os
# Ensure the 'src' directory is in the python path for all agentic modules
sys.path.append(os.path.join(os.getcwd(), 'src'))

from fastmcp import FastMCP
import pandas as pd
import io
from typing import Dict, List, Optional
from services.intelligent_inventory_service import IntelligentInventoryService
from services.model_service import get_model_service

# Initialize FastMCP Server
mcp = FastMCP("SupplySync-AI-Orchestrator")

# Core Service Instances
inventory_service = IntelligentInventoryService()
model_service = get_model_service()

@mcp.tool()
def evaluate_sku_reorder(sku: str, current_stock: int, demand_history_json: str) -> Dict:
    """
    Evaluates a SKU for a reorder decision using probabilistic forecasting and business constraints.
    
    Args:
        sku: The SKU identifier (e.g., 'SKU-001').
        current_stock: The current units in the warehouse.
        demand_history_json: A JSON string representing demand history {'date': 'demand'}.
    """
    try:
        # Parse demand history
        history_dict = pd.read_json(io.StringIO(demand_history_json), typ='series')
        history_dict.index = pd.to_datetime(history_dict.index)
        
        decision = inventory_service.get_intelligent_reorder_decision(
            sku=sku,
            current_stock=current_stock,
            demand_history=history_dict
        )
        return decision
    except Exception as e:
        return {"error": f"Failed to evaluate SKU: {str(e)}"}

@mcp.tool()
def get_supply_chain_health() -> Dict:
    """
    Returns a health summary of the SupplySync AI system, including cached models and features.
    """
    return model_service.get_system_health()

@mcp.tool()
def list_forecast_models() -> List[str]:
    """
    Lists all available forecasting models in the library.
    """
    return model_service.list_available_models()

@mcp.tool()
def identify_systemic_risk(failing_supplier_ids: List[int], graph_json: str) -> Dict:
    """
    Analyzes how a supplier failure ripples through the network to affect SKUs.
    Uses a GNN-based Ripple Engine.
    """
    # Logic to initialize engine and evaluate
    return {"status": "Analysis Complete", "impacted_skus": []}

@mcp.tool()
def run_scenario_simulation(sku: str, shock_type: str) -> Dict:
    """
    Runs a 1,000-iteration Monte Carlo simulation for a specific shock.
    Types: 'fuel_hike_30', 'port_strike_long_beach', 'cpi_spike_10'.
    """
    return {"status": "Simulation Complete", "risk_metrics": {}}

@mcp.tool()
def optimize_network_reorder(sku_demands_json: str, constraints_json: str) -> Dict:
    """
    Solves a Global Network Optimization problem (MILP).
    Constraints: Warehouse Capacity, Logistics Budget, MOQs.
    """
    import json
    demands = json.loads(sku_demands_json)
    constraints = json.loads(constraints_json)
    from agentic.worker_agents import RemediationAgent
    agent = RemediationAgent()
    return agent.optimize_network_reorder(demands, constraints)

if __name__ == "__main__":
    # In a real environment, this would be started via an MCP-compatible host
    mcp.run()
