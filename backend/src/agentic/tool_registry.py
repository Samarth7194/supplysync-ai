# src/agentic/tool_registry.py

from enum import Enum
from typing import Dict, List, Set, Optional, Callable, Any
import logging

# Define Architectural Roles for the Multi-Agent System (MAS)
class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    TRIAGE = "triage"
    ENRICHMENT = "enrichment"
    REMEDIATION = "remediation"

# Define Permission Boundaries
class Permission(str, Enum):
    READ_INVENTORY = "read_inventory"
    READ_FORECAST = "read_forecast"
    WRITE_INVENTORY = "write_inventory"
    EXECUTE_PAYMENT = "execute_payment"
    WEB_BROWSER_ACCESS = "web_browser_access"
    RESERVE_WAREHOUSE_CAPACITY = "reserve_warehouse_capacity"

class ToolRegistry:
    """
    Lead Architect Implementation: A guardrailed registry that enforces 
    boundary-based access control (BBAC) for autonomous supply chain agents.
    """
    
    def __init__(self):
        self._logger = logging.getLogger("tool_registry")
        # Define role-to-permission mapping (Immutable in production)
        self._registry: Dict[AgentRole, Set[Permission]] = {
            AgentRole.SUPERVISOR: {
                Permission.READ_INVENTORY, 
                Permission.READ_FORECAST, 
                Permission.WRITE_INVENTORY,
                Permission.EXECUTE_PAYMENT,
                Permission.RESERVE_WAREHOUSE_CAPACITY,
                Permission.WEB_BROWSER_ACCESS
            },
            AgentRole.TRIAGE: {
                Permission.READ_INVENTORY, 
                Permission.READ_FORECAST
            },
            AgentRole.ENRICHMENT: {
                Permission.READ_INVENTORY, 
                Permission.WEB_BROWSER_ACCESS
            },
            AgentRole.REMEDIATION: {
                Permission.READ_INVENTORY, 
                Permission.WRITE_INVENTORY,
                Permission.RESERVE_WAREHOUSE_CAPACITY
            }
        }
        
    def validate_access(self, role: AgentRole, action: Permission) -> bool:
        """
        Validates if an agent has the authority to execute a specific tool action.
        
        Args:
            role: The role requesting the action.
            action: The sensitive API permission required.
        """
        allowed_actions = self._registry.get(role, set())
        has_access = action in allowed_actions
        
        if not has_access:
            self._logger.warning(f"SECURITY ALERT: Blocked unauthorized action '{action}' from agent role '{role}'")
        
        return has_access

    def get_role_capabilities(self, role: AgentRole) -> List[Permission]:
        """Provides a capability list for an agent during its 'Plan' phase."""
        return list(self._registry.get(role, set()))

# Global Registry Instance
_global_registry = ToolRegistry()

def get_tool_registry():
    """Returns the SupplySync Tool Registry for agent guardrailing."""
    return _global_registry

# Design Pattern: Tool Wrapper with Guardrail Decoration
def guarded_tool(role: AgentRole, permission: Permission):
    """Decorator to enforce tool-level isolation for worker agents."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            if not _global_registry.validate_access(role, permission):
                raise PermissionError(f"Access Denied: Role '{role}' cannot perform '{permission}'")
            return func(*args, **kwargs)
        return wrapper
    return decorator
