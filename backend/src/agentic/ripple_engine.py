# src/agentic/ripple_engine.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import numpy as np

class RippleNetworkGNN(nn.Module):
    """
    Lead Architect Implementation: Graph Neural Network (GNN) for 
    systemic risk propagation. 
    
    Predicts how a failure in one node (e.g., Supplier) ripples through 
    the dependency graph to affect SKU nodes.
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        super(RippleNetworkGNN, self).__init__()
        # Use GraphConv to capture edge-weighted dependencies
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, 1) # Final Risk Score (0-1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Propagates risk scores across the network.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Adjacency list [2, num_edges]
            edge_weight: Strength of dependency [num_edges]
        """
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).sigmoid()
        return x

class SystemicRippleEngine:
    """
    Orchestrates the GNN for supply chain resilience analysis.
    Moves beyond SKU-level logic into Network-level dependency modeling.
    
    Task 5.1: Dependency Ripple Engine.
    """
    
    def __init__(self, num_skus: int, num_suppliers: int):
        self.num_skus = num_skus
        self.num_suppliers = num_suppliers
        self.model = RippleNetworkGNN()
        
    def identify_systemic_impact(
        self, 
        failing_supplier_ids: List[int], 
        node_features: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Predicts which SKUs will be impacted by a supplier failure.
        
        Args:
            failing_supplier_ids: Indices of the failing nodes.
            node_features: State features (lead time, criticality, stock).
            edge_index: Graph topology.
        """
        self.model.eval()
        with torch.no_grad():
            # Seed the failing nodes with maximum risk (1.0)
            # node_features[:, 0] == Risk Seed
            node_features[failing_supplier_ids, 0] = 1.0
            
            risk_scores = self.model(node_features, edge_index)
            
            # Map indices back to SKUs
            impacted_skus = []
            for i in range(self.num_suppliers, self.num_skus + self.num_suppliers):
                if risk_scores[i] > 0.4: # Risk Threshold
                    impacted_skus.append({
                        "id": i,
                        "risk_score": float(risk_scores[i]),
                        "criticality": "HIGH" if risk_scores[i] > 0.8 else "MEDIUM"
                    })
                    
            return {
                "failing_suppliers": failing_supplier_ids,
                "impacted_skus_count": len(impacted_skus),
                "impacted_details": impacted_skus,
                "systemic_risk_index": float(risk_scores.mean())
            }

# Design Pattern: Triage Agent Integration
def enrich_triage_with_ripple(triage_results: Dict, ripple_engine: SystemicRippleEngine, graph_state: Dict) -> Dict:
    """
    Adds systemic risk context to base triage results.
    Bridges the gap between SKU-level triage and Network-level propagation.
    """
    sku_id = triage_results.get("sku")
    
    # If a SKU is at high risk, treat its primary suppliers as potential fail points 
    # for the systemic analysis to see what ELSE might fail.
    mock_suppliers = [0, 1] # In production, these are looked up from the ontology
    
    ripple_impact = ripple_engine.identify_systemic_impact(
        failing_supplier_ids=mock_suppliers,
        node_features=graph_state['features'],
        edge_index=graph_state['edge_index']
    )
    
    triage_results["systemic_context"] = {
        "ripple_risk_index": ripple_impact["systemic_risk_index"],
        "potentially_impacted_skus": ripple_impact["impacted_skus_count"],
        "is_systemic_threat": ripple_impact["systemic_risk_index"] > 0.6
    }
    
    return triage_results
