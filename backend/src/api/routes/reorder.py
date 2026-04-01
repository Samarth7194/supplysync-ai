# src/api/routes/reorder.py

from fastapi import APIRouter, HTTPException
from api.schemas import ReorderRequest, ReorderResponse
from services.model_service import get_model_service
from services.intelligent_inventory_service import IntelligentInventoryService
from services.data_service import DataService
from inventory.business_constraints import SupplierConstraints, get_default_supplier_constraints

router = APIRouter()


@router.post("/reorder", response_model=ReorderResponse)
def reorder(request: ReorderRequest):
    """Returns intelligent reorder recommendation with real data and business constraints."""
    try:
        model_service = get_model_service()

        try:
            model = model_service.load_model("lightgbm_demand_forecast")
        except FileNotFoundError:
            model = None

        # Load real demand history
        try:
            data_service = DataService.get_instance()
            demand_history = data_service.get_demand_history(request.sku).tail(60)
            if len(demand_history) < 7:
                raise HTTPException(status_code=404, detail=f"Insufficient data for SKU {request.sku}")
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Data not available. Run train_model.py first.")

        last_features = model_service.get_cached_features(request.sku)
        supplier_constraints = get_default_supplier_constraints().get("regular", SupplierConstraints())

        intelligent_service = IntelligentInventoryService(model=model)
        decision = intelligent_service.get_intelligent_reorder_decision(
            sku=request.sku,
            current_stock=request.current_stock,
            demand_history=demand_history,
            lead_time_days=request.lead_time_days,
            last_features=last_features,
            supplier_constraints=supplier_constraints,
        )

        model_service.log_decision(decision, {"endpoint": "reorder_api"})

        return {
            "sku": decision["sku"],
            "action": decision["action"],
            "order_quantity": decision["order_quantity"],
            "reorder_point": decision["reorder_point"],
            "safety_stock": decision["safety_stock"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reorder decision failed: {str(e)}")
