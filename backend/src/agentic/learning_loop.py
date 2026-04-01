# src/agentic/learning_loop.py

from typing import Dict, Any, List, Optional
import numpy as np

class CausalFeedbackLoop:
    """
    Lead Architect Implementation: The Self-Healing Feedback Loop.
    Captures human overrides and recalibrates agent parameters using 
    Causal Inference principles.
    
    Task 7.2: Self-Healing Feedback Loop.
    """
    
    def __init__(self, current_lambda: float = 0.2):
        self.risk_lambda = current_lambda
        self.learning_rate = 0.05
        self._veto_history: List[Dict] = []

    def log_veto(self, original_qty: int, human_qty: int, reason: str):
        """
        Learns from a human rejection of an autonomous decision.
        """
        delta_ratio = (original_qty - human_qty) / original_qty if original_qty > 0 else 0
        
        veto_data = {
            "original_qty": original_qty,
            "human_qty": human_qty,
            "delta": delta_ratio,
            "reason": reason
        }
        self._veto_history.append(veto_data)
        
        # Self-Healing Logic: Trigger recalibration if delta > 20%
        if abs(delta_ratio) > 0.2:
            self.recalibrate_hyperparameters(delta_ratio)

    def recalibrate_hyperparameters(self, delta_ratio: float):
        """
        Adjusts the risk-aversion parameter (lambda) based on human veto.
        If human reduces order, agent was too risk-averse (needs lower lambda).
        """
        adjustment = delta_ratio * self.learning_rate
        old_lambda = self.risk_lambda
        self.risk_lambda = max(0.01, self.risk_lambda - adjustment)
        
        print(f"SELF-HEALING: Risk Lambda shifted from {old_lambda:.4f} to {self.risk_lambda:.4f} based on human feedback.")

    def get_calibration_report(self) -> Dict[str, Any]:
        return {
            "current_lambda": self.risk_lambda,
            "total_vetoes_processed": len(self._veto_history),
            "avg_override_delta": np.mean([v['delta'] for v in self._veto_history]) if self._veto_history else 0
        }
