# src/api/routes/reorder.py

from fastapi import APIRouter, HTTPException
from src.api.schemas import ReorderRequest, ReorderResponse
from src.services.model_service import get_model_service
from src.services.intelligent_inventory_service import IntelligentInventoryService
from src.inventory.business_constraints import SupplierConstraints, get_default_supplier_constraints
import pandas as pd
import numpy as np

router = APIRouter()

@router.post("/reorder", response_model=ReorderResponse)
def reorder(request: ReorderRequest):
    """
    Returns intelligent reorder recommendation for a SKU with adaptive forecasting and business constraints.
    """
    
    try:
        # Get model service
        model_service = get_model_service()
        
        # Load trained model
        try:
            model = model_service.load_model("lightgbm_demand_forecast")
        except FileNotFoundError:
            # Continue without ML model - will use adaptive forecasting with fallbacks
            model = None
        
        # Initialize intelligent service
        intelligent_service = IntelligentInventoryService(model=model)
        
        # Create demand history (in production, this would come from database)
        # For demo purposes, generate realistic demand pattern
        np.random.seed(hash(request.sku) % 1000)  # Reproducible per SKU
        demand_history = pd.Series(np.random.poisson(8, 60))  # 60 days of history
        
        # Get cached features if available
        last_features = model_service.get_cached_features(request.sku)
        
        # Get supplier constraints (in production, this would come from supplier database)
        supplier_constraints = get_default_supplier_constraints().get("regular", SupplierConstraints())
        
        # Get intelligent reorder decision
        decision = intelligent_service.get_intelligent_reorder_decision(
            sku=request.sku,
            current_stock=request.current_stock,
            demand_history=demand_history,
            lead_time_days=request.lead_time_days,
            last_features=last_features,
            supplier_constraints=supplier_constraints
        )
        
        # Log the decision
        model_service.log_decision(decision, {"endpoint": "reorder_api"})
        
        return {
            "sku": decision["sku"],
            "action": decision["action"],
            "order_quantity": decision["order_quantity"],
            "reorder_point": decision["reorder_point"],
            "safety_stock": decision["safety_stock"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reorder decision failed: {str(e)}")
