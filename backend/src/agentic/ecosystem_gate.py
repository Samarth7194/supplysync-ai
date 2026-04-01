# src/agentic/ecosystem_gate.py

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder

class EcosystemGate:
    """
    Lead Architect Implementation: Agent-to-Agent (A2A) Protocol.
    Enables autonomous business negotiation with cryptographic non-repudiation.
    
    Task 8.1: A2A Negotiation Protocol with ED25519.
    """
    
    def __init__(self, agent_uri: str = "uri:agent:supplysync-01"):
        self.agent_uri = agent_uri
        self.logger = logging.getLogger("ecosystem_gate")
        
        # Generate identity keys for signing
        # In production, these would be loaded from a secure vault or HSM
        self._signing_key = SigningKey.generate()
        self.verify_key = self._signing_key.verify_key

    def create_proposal(self, target_uri: str, intent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a signed Order Proposal following the A2A schema.
        """
        payload = {
            "intent": intent,
            "data": data,
            "recourse_options": [
                {"type": "LEAD_TIME_TRADE_OFF", "delta": "+5d", "discount": "5%"},
                {"type": "PARTIAL_FULFILLMENT", "min_qty": 50}
            ]
        }
        
        header = {
            "protocol_version": "v1.0",
            "timestamp": datetime.now().isoformat(),
            "sender_uri": self.agent_uri,
            "target_uri": target_uri
        }
        
        # Canonicalize and sign the payload
        message_to_sign = json.dumps({"header": header, "payload": payload}, sort_keys=True).encode("utf-8")
        signature = self._signing_key.sign(message_to_sign).signature
        
        return {
            "header": header,
            "payload": payload,
            "signature": HexEncoder.encode(signature).decode("utf-8"),
            "verify_key": HexEncoder.encode(self.verify_key.encode()).decode("utf-8")
        }

    def verify_message(self, message: Dict[str, Any]) -> bool:
        """
        Verifies the cryptographic integrity of an inbound A2A message.
        """
        try:
            sig = HexEncoder.decode(message["signature"])
            v_key_hex = HexEncoder.decode(message["verify_key"])
            verify_key = VerifyKey(v_key_hex)
            
            # Reconstruct the original signed string
            original_data = json.dumps({
                "header": message["header"], 
                "payload": message["payload"]
            }, sort_keys=True).encode("utf-8")
            
            verify_key.verify(original_data, sig)
            return True
        except Exception as e:
            self.logger.error(f"A2A Signature Verification Failed: {str(e)}")
            return False

    def negotiate_order(self, target_agent_uri: str, data: Dict) -> Dict[str, Any]:
        """
        Orchestrates an autonomous negotiation session.
        """
        proposal = self.create_proposal(target_agent_uri, "ORDER_PROPOSAL", data)
        self.logger.info(f"A2A_SEND: Initial Proposal Sent to {target_agent_uri}")
        
        # Interaction with external agents would happen via MCP or HTTP here
        # For the build, we call the mock supplier directly
        from agentic.mock_supplier import MockSupplierAgent
        supplier = MockSupplierAgent(target_agent_uri)
        
        response = supplier.receive_message(proposal)
        
        if self.verify_message(response):
            self.logger.info(f"A2A_RECV: Verified Response Intent: {response['payload']['intent']}")
            return response
        else:
            return {"error": "Sec_Auth_Failure"}

# Design Pattern: Cryptographic Handshake Hook
def initiate_secure_handshake(orchestrator_uri: str, supplier_uri: str) -> Dict:
    gate = EcosystemGate(orchestrator_uri)
    return gate.negotiate_order(supplier_uri, {"sku": "GPU-A100", "requested_qty": 200})
