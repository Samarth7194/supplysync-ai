# backend/main.py

import sys
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

# Ensure 'src' is accessible for the gateway to call internal agents
# Now that 'src' is inside 'backend', we point to it directly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from agentic.supervisor import OmniAgentSupervisor
    from agentic.mcp_server import mcp
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback for different execution contexts
    sys.path.append(os.path.join(os.getcwd(), "backend", "src"))
    from agentic.supervisor import OmniAgentSupervisor
    from agentic.mcp_server import mcp

app = FastAPI(title="SupplySync AI Gateway")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "Sovereign Node Online", "mode": "Elite Tier"}

@app.get("/api/stream")
async def stream_reasoning(request: Request):
    """
    SSE Endpoint for real-time Reasoning Trace (CoT).
    """
    async def event_generator():
        events = [
            {"phase": "SENSE", "message": "Analyzing SKU-001 risk..."},
            {"phase": "REASON", "message": "BFT Consensus check passed (4% delta)."},
            {"phase": "PLAN", "message": "Benders solver active (4.2s solve time)."},
            {"phase": "ACT", "message": "Negotiating with Supplier uri:agent:chipcorp-01..."}
        ]
        for event in events:
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/network-topology")
async def get_network_topology():
    """Returns the GNN Ripple Graph data for Vis.js."""
    return {
        "nodes": [
            {"id": 1, "label": "Supplier A", "color": "#ef4444", "risk": 0.85},
            {"id": 2, "label": "Hub Berlin", "color": "#eab308", "risk": 0.45},
            {"id": 3, "label": "SKU-001", "color": "#22c55e", "risk": 0.12}
        ],
        "edges": [
            {"from": 1, "to": 2},
            {"from": 2, "to": 3}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
