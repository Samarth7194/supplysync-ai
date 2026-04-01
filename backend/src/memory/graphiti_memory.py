# src/memory/graphity_memory.py

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from graphiti import Graphiti
from graphiti.edges import TemporalRelationship
from graphiti.nodes import SKU, Supplier, Warehouse

class BiTemporalMemoryManager:
    """
    Lead Architect Implementation: Bi-Temporal Knowledge Graph using Graphiti. 
    This layer enables reasoning over the "history of truth" by tracking 
    both VALID_TIME (when events occurred) and SYSTEM_TIME (when ingested).
    
    Task 3.1 & 3.2: Deploy Graphiti for Bi-Temporal Modeling.
    """
    
    def __init__(self, connection_string: str = "bolt://localhost:7687"):
        self.graph = Graphiti(connection_string)
        self.logger = logging.getLogger("graphiti_memory")
        
    def add_factual_assertion(
        self, 
        entity_id: str, 
        entity_type: str, 
        assertion: str, 
        valid_at: datetime
    ) -> bool:
        """
        Task 3.3: Incremental Memory Updates.
        Extracts factual assertions and updates the graph with bi-temporal edges.
        
        Args:
            entity_id: Target entity identifier (e.g., SKU-001)
            entity_type: SKU, Supplier, or Warehouse
            assertion: The fact being recorded (e.g., "Supplier delayed by 2 days")
            valid_at: When the event actually occurred in the real world
        """
        try:
            # Task 3.2: Validity intervals [t_valid, t_invalid]
            self.graph.add_edge_temporal(
                source=entity_id,
                target=entity_type,
                fact=assertion,
                valid_from=valid_at,
                valid_to=None # None implies current state of truth
            )
            return True
        except Exception as e:
            self.logger.error(f"Memory update failed: {e}")
            return False

    def state_reasoning_query(self, query: str, timestamp: datetime) -> List[Dict]:
        """
        Ask the memory "What was the truth at time T?"
        
        Task 3.4: Hybrid Retrieval Engine (Semantic + BM25 + Graph Traversal)
        """
        # Architectural Decision: Combine semantic graph traversal with 
        # temporal filters to resolve the state of truth at 'timestamp'.
        results = self.graph.search(
            query=query, 
            at_time=timestamp,
            top_k=5,
            search_method="hybrid" # Task 3.4
        )
        return results

    def get_supplier_delay_history(self, supplier_id: str) -> List[Dict]:
        """Specific retrieval for remediation agent reasoning."""
        return self.graph.get_temporal_history(
            node_id=supplier_id, 
            relationship_type="DELAYED_BY"
        )
