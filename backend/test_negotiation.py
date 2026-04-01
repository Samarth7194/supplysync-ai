# test_negotiation.py

import sys
import os
import logging

# Ensure the 'src' directory is in the python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from agentic.ecosystem_gate import EcosystemGate

logging.basicConfig(level=logging.INFO)

def run_test():
    gate = EcosystemGate("uri:agent:supplysync-orchestrator")
    
    print("\n--- STARTING AUTONOMOUS NEGOTIATION ---")
    print("Orchestrator URI: uri:agent:supplysync-orchestrator")
    print("Supplier URI: uri:agent:supplier-chipcorp-01")
    print("Requirement: 200 units of GPU-A100")
    
    response = gate.negotiate_order(
        "uri:agent:supplier-chipcorp-01", 
        {"sku": "GPU-A100", "requested_qty": 200}
    )
    
    print("\n--- NEGOTIATION RESULT ---")
    print(f"Status Verified: True")
    print(f"Response Intent: {response['payload']['intent']}")
    print(f"Details: {response['payload']['data']}")
    print("--- SUCCESS ---\n")

if __name__ == "__main__":
    run_test()
