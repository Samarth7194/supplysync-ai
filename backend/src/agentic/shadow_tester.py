# src/agentic/shadow_tester.py

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from services.intelligent_inventory_service import IntelligentInventoryService

class ShadowTestingModule:
    """
    Lead Architect Implementation: Shadow Testing guardrail.
    Runs agentic decisions in parallel with the legacy deterministic system 
    for 48 hours to validate safety before execution.
    
    Task 4.3: Shadow Testing Workflow.
    """
    
    def __init__(self, legacy_service: IntelligentInventoryService):
        self.legacy_service = legacy_service
        self.logger = logging.getLogger("shadow_tester")
        self._shadow_logs: List[Dict] = []
        
    def execute_and_log_parallel(
        self, 
        sku: str, 
        agent_decision: Dict, 
        input_context: Dict
    ) -> Dict:
        """
        Runs the legacy logic in shadow mode for the same SKU and input data.
        
        Returns:
            Comparison metadata between agentic and deterministic logic.
        """
        # Run Legacy Logic (Deterministic)
        legacy_decision = self.legacy_service.get_intelligent_reorder_decision(
            sku=sku,
            current_stock=input_context['current_stock'],
            demand_history=input_context['demand_history']
        )
        
        # Extract Quantities for comparison
        # Agent remediation format: {'remediation_plan': {'final_quantity': 100}}
        agent_qty = agent_decision.get("remediation_plan", {}).get("final_quantity", 0)
        legacy_qty = legacy_decision.get("order_quantity", 0)
        
        # Calculate Deviation Metric
        deviation_ratio = 0.0
        if legacy_qty > 0:
            deviation_ratio = abs(agent_qty - legacy_qty) / legacy_qty
            
        is_critical = deviation_ratio > 0.5 # 50% threshold for critical deviation
        
        from infra.observability import HueObservability
        obs = HueObservability()
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "sku": sku,
            "agent_reorder": agent_qty,
            "legacy_reorder": legacy_qty,
            "deviation_ratio": round(deviation_ratio, 4),
            "is_critical": is_critical,
            "agent_logic": agent_decision.get("status", "COMPLETE"),
            "legacy_logic": legacy_decision.get("explainability", {}).get("decision_logic"),
            "approval_required": is_critical # Flag for Task 4.3 Human Review
        }
        
        # Log to Production Observability
        obs.log_decision_metrics(
            sku=sku, 
            latency_ms=agent_decision.get("metadata", {}).get("latency", 150),
            entropy=agent_decision.get("metadata", {}).get("entropy", 0.05),
            throughput_tokens=agent_decision.get("metadata", {}).get("tokens", 450)
        )
        
        if is_critical:
            self.logger.critical(f"SHADOW REJECTION: SKU {sku} - Agentic logic deviated by {deviation_ratio*100}% from legacy baseline.")
            
        self._shadow_logs.append(comparison)
        return comparison

    def get_summary_report(self) -> Dict:
        """Task 4.3: Summary of the 48-hour shadow period."""
        total = len(self._shadow_logs)
        critical = sum(1 for log in self._shadow_logs if log['is_critical'])
        
        return {
            "total_decisions_logged": total,
            "critical_discrepancies": critical,
            "safety_pass_rate": round((total - critical) / total if total > 0 else 1.0, 3)
        }
