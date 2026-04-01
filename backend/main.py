# backend/main.py

import sys
import os
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Ensure 'src' is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from services.intelligent_inventory_service import IntelligentInventoryService
from services.model_service import get_model_service
from ingestion.load_retail_data import get_sku_descriptions

app = FastAPI(title="SupplySync AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global references (loaded on startup)
_loaded_model = None
_data_service = None
_sku_descriptions = {}


@app.on_event("startup")
def load_models():
    global _loaded_model, _data_service, _sku_descriptions
    # Load trained model
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    model_service = get_model_service(model_dir=saved_models_dir)
    try:
        _loaded_model = model_service.load_model("lightgbm_demand_forecast")
    except FileNotFoundError:
        _loaded_model = None

    # Load data service
    try:
        from services.data_service import DataService
        _data_service = DataService.get_instance()
    except Exception:
        _data_service = None

    # Cache SKU descriptions
    try:
        _sku_descriptions = get_sku_descriptions()
    except Exception:
        _sku_descriptions = {}


@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": _loaded_model is not None}


@app.get("/api/kpis")
async def kpis():
    """Return cached KPIs from simulation."""
    cache_path = Path(__file__).resolve().parent / "data" / "cached_kpis.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {"error": "KPIs not computed. Run: python scripts/compute_kpis.py"}


@app.get("/api/skus")
async def list_skus():
    """Return top SKUs available for analysis."""
    if _data_service is None:
        return {"skus": []}
    top = _data_service.get_top_skus(n=20)
    return {"skus": top}


@app.get("/api/skus/details")
async def skus_with_details():
    """Return top SKUs with names and demand statistics."""
    if _data_service is None:
        return {"skus": []}
    top_skus = _data_service.get_top_skus(n=20)
    result = []
    for sku_code in top_skus:
        demand = _data_service.get_demand_history(sku_code)
        avg_demand = float(demand.mean()) if len(demand) > 0 else 0
        total_demand = float(demand.sum()) if len(demand) > 0 else 0
        result.append({
            "id": sku_code,
            "name": _sku_descriptions.get(sku_code, f"SKU {sku_code}"),
            "avg_demand": round(avg_demand, 1),
            "total_demand": round(total_demand),
        })
    return {"skus": result}


@app.post("/api/analyze")
async def analyze_sku(request: Request):
    """Analyze a SKU and return risk assessment + reorder recommendation."""
    try:
        body = await request.json()
        sku = body.get("sku", "85123A")
        current_stock = body.get("current_stock", 50)
        demand_history_input = body.get("demand_history", [])

        # Use real data if available and no history provided
        if not demand_history_input and _data_service is not None:
            demand_series = _data_service.get_demand_history(sku)
            if len(demand_series) > 0:
                demand_history_input = demand_series.tail(60).tolist()

        # Fallback to random if no data
        if not demand_history_input:
            np.random.seed(hash(sku) % 1000)
            demand_history_input = list(np.random.poisson(20, 30))

        # Use intelligent service for real decisions
        import pandas as pd
        intelligent_service = IntelligentInventoryService(model=_loaded_model)
        decision = intelligent_service.get_intelligent_reorder_decision(
            sku=sku,
            current_stock=current_stock,
            demand_history=pd.Series(demand_history_input),
            lead_time_days=7,
        )

        # Compute p50/p90 from demand history for risk classification
        p50 = float(np.mean(demand_history_input))
        p90 = float(np.percentile(demand_history_input, 90))

        if current_stock < p50:
            risk, risk_color = "HIGH", "#ef4444"
        elif current_stock < p90:
            risk, risk_color = "MEDIUM", "#eab308"
        else:
            risk, risk_color = "LOW", "#22c55e"

        order_qty = decision.get("order_quantity", 0)

        return {
            "sku": sku,
            "risk": risk,
            "risk_color": risk_color,
            "forecast": {"p50": round(p50, 1), "p90": round(p90, 1)},
            "current_stock": current_stock,
            "recommended_order": order_qty,
            "action": "PURCHASE" if order_qty > 0 else "NO_ACTION",
            "demand_pattern": decision.get("intelligence", {}).get("demand_pattern", "unknown"),
            "forecast_method": decision.get("intelligence", {}).get("forecast_method", "unknown"),
        }

    except Exception as e:
        return {"error": str(e), "sku": "unknown", "status": "FAILED"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
