# src/infra/observability.py

import time
import logging
from typing import Dict, Any, List
from datetime import datetime

class HueObservability:
    """
    Lead Architect Implementation: Hue-style observability.
    Tracks operational metrics (latency, throughput) and AI-specific indicators.
    """
    
    def __init__(self):
        self.metrics: List[Dict] = []
        self.logger = logging.getLogger("hue_observability")
        
    def log_decision_metrics(self, sku: str, latency_ms: float, entropy: float, throughput_tokens: int):
        """
        Logs production-grade metrics for a single agentic decision.
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "sku": sku,
            "latency_ms": latency_ms,
            "prediction_entropy": entropy, # High entropy = Low confidence
            "token_throughput": throughput_tokens,
            "status": "HEALTHY" if latency_ms < 500 else "DEGRADED"
        }
        self.metrics.append(metric)
        self.logger.info(f"HUE_METRIC: SKU={sku} LATENCY={latency_ms}ms ENTROPY={entropy}")

    def get_aggregate_health(self) -> Dict[str, Any]:
        """Returns avg latency and failure rates for SRE dashboards."""
        if not self.metrics:
            return {"status": "NO_DATA"}
            
        avg_latency = sum(m['latency_ms'] for m in self.metrics) / len(self.metrics)
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "total_calls": len(self.metrics),
            "critical_latency_breaches": sum(1 for m in self.metrics if m['latency_ms'] > 1000)
        }
