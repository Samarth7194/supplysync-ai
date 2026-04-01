# src/infra/ontology.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class EntityType(Enum):
    PRODUCT = "Product"
    SUPPLIER = "Supplier"
    HUB = "Hub"
    ORDER = "Order"
    SHIPMENT = "Shipment"
    RECOURSE = "RecourseAction" # Phase 7
    VETO = "VetoReason" # Phase 7

@dataclass
class OntologyNode:
    """
    Lead Architect Implementation: Unified Semantic Layer.
    Base class for RDF-compliant entity modeling.
    """
    uri: str
    entity_type: EntityType
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid_from: datetime = field(default_factory=datetime.now)

@dataclass
class Product(OntologyNode):
    sku: str = ""
    uom: str = "each"
    criticality_score: float = 0.5

@dataclass
class RecourseAction(OntologyNode):
    associated_scenario: str = ""
    remedy_type: str = "EXPEDITED_FREIGHT"
    estimated_cost: float = 0.0

@dataclass
class VetoReason(OntologyNode):
    override_delta: float = 0.0
    human_comment: str = ""
    inferred_cause: str = "CalibrationError"

class OperationalOntology:
    """
    Manages the entity relationships (Supplier -> Hub -> SKU)
    based on Netflix's Unified Data Architecture (UDA).
    """
    
    def __init__(self):
        self._store: Dict[str, OntologyNode] = {}
        self._relations: List[Dict] = []
        
    def add_entity(self, entity: OntologyNode):
        self._store[entity.uri] = entity
        
    def add_relation(self, subject_uri: str, predicate: str, object_uri: str):
        """Simplistic RDF-triple representation."""
        self._relations.append({
            "s": subject_uri,
            "p": predicate,
            "o": object_uri
        })

    def resolve_business_object(self, uri: str) -> Optional[OntologyNode]:
        """Resolves raw IDs into coherent business objects."""
        return self._store.get(uri)

# Design Pattern: Knowledge-Graph-First Orchestration
def get_global_ontology() -> OperationalOntology:
    """Singleton for the operational semantic layer."""
    ontology = OperationalOntology()
    # Mock data for demonstration
    s1 = Supplier("uri:supplier:001", EntityType.SUPPLIER, {"name": "GlobalChips Inc"}, 0)
    p1 = Product("uri:product:GPU-A100", EntityType.PRODUCT, {"sku": "GPU-A100"}, 0)
    
    ontology.add_entity(s1)
    ontology.add_entity(p1)
    ontology.add_relation(s1.uri, "SUPPLIES", p1.uri)
    
    return ontology
