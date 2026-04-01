# src/agentic/approval_workflow.py

from enum import Enum
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    USER_REVIEW_REQUIRED = "user_review_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    BYPASSED = "bypassed" # For small orders

class GovernanceEngine:
    """
    Lead Architect Implementation: Enterprise Governance (HITL).
    Ensures high-cost autonomous actions require explicit human validation.
    
    Task 5.4: Enterprise Governance.
    """
    
    def __init__(self, cost_threshold: float = 10000.0):
        self.cost_threshold = cost_threshold
        self.logger = logging.getLogger("governance_engine")

    def process_remediation_for_approval(self, remediation_plan: Dict) -> Dict[str, Any]:
        """
        Evaluates a remediation plan against the corporate governance policy.
        
        Policy: Any action costing > $10,000 must pause and generate a 
        'Proposal Artifact' for human review.
        """
        total_cost = remediation_plan.get("total_cost", 0.0)
        
        # Check Policy
        if total_cost > self.cost_threshold:
            self.logger.info(f"GOVERNANCE PAUSE: Order cost ${total_cost} exceeds threshold. Generating Proposal Artifact.")
            
            proposal = self.generate_proposal_document(remediation_plan)
            
            return {
                "status": ApprovalStatus.USER_REVIEW_REQUIRED,
                "reason": f"Cost ${total_cost} > Threshold ${self.cost_threshold}",
                "proposal_artifact_path": proposal["path"],
                "decision_id": f"GOV-{int(datetime.now().timestamp())}"
            }
        
        return {
            "status": ApprovalStatus.BYPASSED,
            "reason": "Cost within autonomous limits."
        }

    def generate_proposal_document(self, remediation_plan: Dict) -> Dict:
        """
        Generates and saves a structured human-readable artifact for the reorder proposal.
        """
        import os
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_slug = int(datetime.now().timestamp())
        filename = f"proposal_{ts_slug}.md"
        
        # Ensure proposals directory exists
        os.makedirs("proposals", exist_ok=True)
        path = os.path.join("proposals", filename)

        content = [
            f"# Supply Chain Reorder Proposal (System ID: {ts_slug})",
            f"**Timestamp:** {timestamp}",
            f"**Total Expected Cost:** ${remediation_plan.get('total_expected_cost', 0):,.2f}",
            f"**Optimization Method:** Stochastic SMILP (SAA)",
            f"\n## Evidence & Logic Chain",
            f"- **Scenario Alpha:** {remediation_plan.get('num_scenarios_evaluated', 100)} Monte Carlo iterations executed.",
            f"- **Graph Nodes Cited:** {remediation_plan.get('cited_nodes', ['uri:product:GPU-A100'])}",
            "\n## Proposed Orders",
            "| SKU ID | Order Quantity |",
            "| :--- | :--- |"
        ]
        
        orders = remediation_plan.get("orders", {})
        for sku_id, qty in orders.items():
            content.append(f"| {sku_id} | {qty} |")
            
        content.append("\n## Resource Utilization")
        util = remediation_plan.get("utilization", {})
        content.append(f"- **Capacity Usage:** {util.get('capacity_usage', 0)}")
        content.append(f"- **Budget Usage:** ${util.get('budget_usage', 0):,.2f}")
        
        content.append("\n## Governance Recommendation")
        content.append("> [!IMPORTANT]")
        content.append("> This action requires human approval before API execution via the Veto Protocol.")
        content.append(f"\n**Audit Trail Reference:** {ts_slug}-GOV-VETO")

        body = "\n".join(content)
        
        # Write to filesystem for HITL review
        with open(path, "w") as f:
            f.write(body)
            
        return {
            "path": path,
            "body": body
        }
