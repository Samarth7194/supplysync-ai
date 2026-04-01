# src/agentic/mock_supplier.py

import json
from typing import Dict, Any
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder

class MockSupplierAgent:
    """
    Lead Architect Implementation: Mock Supplier Ecosystem Node.
    Responds to A2A proposals with counter-offers to test agentic negotiation logic.
    """
    
    def __init__(self, uri: str = "uri:agent:supplier-chipcorp-01"):
        self.uri = uri
        self._signing_key = SigningKey.generate()
        self.verify_key = self._signing_key.verify_key

    def receive_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a proposal and generates a signed counter-offer.
        """
        intent = message["payload"]["intent"]
        data = message["payload"]["data"]
        
        if intent == "ORDER_PROPOSAL":
            # Simulate a capacity constraint for requested quantities > 150
            if data.get("requested_qty", 0) > 150:
                response_payload = {
                    "intent": "NEGOTIATE_PARTIAL",
                    "data": {
                        "sku": data["sku"],
                        "available_qty": 150,
                        "lead_time": "19d",
                        "status": "Capacity_Restricted"
                    }
                }
            else:
                response_payload = {
                    "intent": "SETTLEMENT",
                    "data": {
                        "sku": data["sku"],
                        "status": "Accepted",
                        "delivery_window": "14d"
                    }
                }
        else:
            response_payload = {"intent": "ACKNOWLEDGE", "data": {"status": "Received"}}

        # Wrap in A2A Schema with signature
        header = {
            "protocol_version": "v1.0",
            "timestamp": message["header"]["timestamp"],
            "sender_uri": self.uri,
            "target_uri": message["header"]["sender_uri"]
        }
        
        msg_to_sign = json.dumps({"header": header, "payload": response_payload}, sort_keys=True).encode("utf-8")
        signature = self._signing_key.sign(msg_to_sign).signature
        
        return {
            "header": header,
            "payload": response_payload,
            "signature": HexEncoder.encode(signature).decode("utf-8"),
            "verify_key": HexEncoder.encode(self.verify_key.encode()).decode("utf-8")
        }
