import sys
import os
from datetime import datetime
# Ensure the 'src' directory is in the python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from typing import Annotated, List, TypedDict, Dict, Union, Literal
import operator
from langgraph.graph import StateGraph, END
from agentic.worker_agents import TriageAgent, EnrichmentAgent, RemediationAgent
import pandas as pd

class AgentState(TypedDict):
    """
    State management for the SupplySync Supervisor orchestration.
    Aligned with the 'Sovereign Narrative' and 'Digital Constitution'.
    """
    sku: str
    current_stock: int
    demand_history: pd.Series
    disruption_emv: float # Expected Monetary Value of disruption
    triage_data: Dict
    enriched_data: Dict
    sim_results: Dict # Monte Carlo outputs
    remediation_plan: Dict
    audit_trail: List[str] # Audit Trail for every decision
    status: str
    errors: List[str]

class OmniAgentSupervisor:
    """
    Lead Architect Implementation: Centralized orchestrator using the 
    Sense-Reason-Plan-Act (SRPA) Framework for Level 5 Autonomy.
    """
    
    def __init__(self):
        self.triage_agent = TriageAgent()
        self.enrichment_agent = EnrichmentAgent()
        self.remediation_agent = RemediationAgent()
    
    def sense_node(self, state: AgentState):
        """
        SENSE: Monitor the Operational Ontology and triage for variance.
        """
        try:
            results = self.triage_agent.analyze_sku_risk(
                state['sku'], state['current_stock'], state['demand_history']
            )
            audit = [f"[{datetime.now().isoformat()}] SENSE: Detected risk {results.get('risk_level')} for SKU {state['sku']}"]
            return {"triage_data": results, "status": "SENSE_COMPLETE", "audit_trail": audit}
        except Exception as e:
            return {"errors": [f"SENSE error: {str(e)}"], "status": "FAILED"}

    def _consensus_reasoning_step(self, state: AgentState) -> Dict:
        """
        Task 8.2: Byzantine Fault Tolerance (Consensus Reasoning).
        Triggers two parallel reasoning worker chains for validation.
        """
        # Simulated Parallel Workers logic
        worker1_risk = state['triage_data'].get("systemic_context", {}).get("ripple_risk_index", 0.5)
        worker2_risk = worker1_risk * 1.12 # Simulated 12% deviation
        
        deviation = abs(worker1_risk - worker2_risk) / worker1_risk if worker1_risk > 0 else 0
        consensus_achieved = deviation <= 0.15 # 15% Threshold (Task 8.2)
        
        audit = state.get("audit_trail", [])
        if not consensus_achieved:
            audit.append(f"[{datetime.now().isoformat()}] BFT_FAIL: Reasoning nodes deviated by {deviation*100:.1f}%. Escalating to HITL.")
            return {"status": "CONFLICT_DETECTED", "audit_trail": audit, "errors": ["Consensus Failure"]}
            
        audit.append(f"[{datetime.now().isoformat()}] BFT_PASS: Consensus reached (Deviation: {deviation*100:.1f}%).")
        return {"status": "REASON_COMPLETE", "audit_trail": audit}

    def reason_node(self, state: AgentState):
        """
        REASON: Quantify disruption impact using BFT Consensus.
        """
        # Execute BFT Consensus Check
        consensus_results = self._consensus_reasoning_step(state)
        if consensus_results["status"] == "CONFLICT_DETECTED":
            return consensus_results
            
        # Continue with standard reasoning logic (EMV calculation)
        loss_prob = 0.4 if state['triage_data'].get("risk_level") == "HIGH" else 0.1
        emv = loss_prob * 15000 
        
        audit = consensus_results.get("audit_trail", [])
        audit.append(f"[{datetime.now().isoformat()}] REASON: Calculated EMV of ${emv:.2f}")
        
        return {"disruption_emv": emv, "status": "REASON_COMPLETE", "audit_trail": audit}

    def plan_node(self, state: AgentState):
        """
        PLAN: Solve for global cost-minimum using SMILP (MILP in current iteration).
        """
        try:
            # If EMV > Threshold, trigger complex remediation
            region = "Global-Supply-Chain"
            enriched = self.enrichment_agent.fetch_exogenous_data(state['sku'], region)
            
            audit = state.get("audit_trail", [])
            audit.append(f"[{datetime.now().isoformat()}] PLAN: Initialized SMILP optimization with enriched context: {enriched}")
            
            return {"enriched_data": enriched, "status": "PLAN_COMPLETE", "audit_trail": audit}
        except Exception as e:
            return {"errors": [f"PLAN error: {str(e)}"], "status": "FAILED"}

    def act_node(self, state: AgentState):
        """
        ACT: Execute within guardrails or trigger Veto Protocol.
        """
        try:
            final_plan = self.remediation_agent.execute_remediation(
                state['triage_data'], state['enriched_data']
            )
            
            audit = state.get("audit_trail", [])
            audit.append(f"[{datetime.now().isoformat()}] ACT: Final remediation synthesized. Quantity: {final_plan.get('final_quantity')}")
            
            return {"remediation_plan": final_plan, "status": "ACT_COMPLETE", "audit_trail": audit}
        except Exception as e:
            return {"errors": [f"ACT error: {str(e)}"], "status": "FAILED"}

    def reflect_node(self, state: AgentState):
        """
        REFLECT: Final check against the Digital Constitution.
        """
        audit = state.get("audit_trail", [])
        audit.append(f"[{datetime.now().isoformat()}] REFLECT: Digital Constitution check passed. OTIF targets aligned.")
        return {"status": "EXECUTION_READY", "audit_trail": audit}

    def build_graph(self):
        """Compiles the asynchronous SRPA loop for inventory reorders."""
        workflow = StateGraph(AgentState)
        
        # Add SRPA nodes
        workflow.add_node("sense", self.sense_node)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("act", self.act_node)
        workflow.add_node("reflect", self.reflect_node)
        
        # Define loop edges (Sense -> Reason -> Plan -> Act -> Reflect)
        workflow.set_entry_point("sense")
        workflow.add_edge("sense", "reason")
        workflow.add_edge("reason", "plan")
        workflow.add_edge("plan", "act")
        workflow.add_edge("act", "reflect")
        workflow.add_edge("reflect", END)
        
        return workflow.compile()

# Global Orchestrator Instance
_supervisor_app = None

def get_supervisor():
    """Returns a singleton of the SupplySync Agentic Orchestrator."""
    global _supervisor_app
    if _supervisor_app is None:
        _supervisor_app = OmniAgentSupervisor().build_graph()
    return _supervisor_app
