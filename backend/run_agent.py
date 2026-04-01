# run_agent.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Start the Supervisor
from src.agentic.supervisor import get_supervisor

def main():
    print("--- SupplySync AI: Agentic Orchestrator Start ---")
    
    # 1. Initialize the Orchestrator (LangGraph Application)
    app = get_supervisor()
    
    # 2. Mock some demand history for the sense phase
    history = pd.Series(
        np.random.randint(5, 50, 30), 
        index=pd.date_range("2024-01-01", periods=30)
    )
    
    # 3. Define the initial state for SKU triage
    # Stock level of 12 will trigger a "High" risk classification
    initial_state = {
        "sku": "GPU-A100", 
        "current_stock": 12, 
        "demand_history": history
    }
    
    print(f"Triggering SRPA loop for SKU: {initial_state['sku']} (Current Stock: {initial_state['current_stock']})")
    
    # 4. Stream the orchestration steps
    try:
        for event in app.stream(initial_state):
            # Extract step name and latest state update
            step_name = list(event.keys())[0]
            state_update = event[step_name]
            
            print(f"\n[PHASE: {step_name.upper()}]")
            if 'status' in state_update:
                print(f"Status: {state_update['status']}")
            
            if 'audit_trail' in state_update:
                print(f"Latest Audit: {state_update['audit_trail'][-1]}")
                
            if 'remediation_plan' in state_update:
                plan = state_update['remediation_plan']
                print(f"Final Quantity Suggested: {plan.get('final_quantity')}")
                print(f"Action: {plan.get('action_type')}")
    except Exception as e:
        print(f"Orchestration failed: {e}")

if __name__ == "__main__":
    main()
