# src/services/intelligent_inventory_service.py

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from services.adaptive_forecasting_service import adaptive_forecast, get_sku_classification_metadata
from inventory.reorder_point import compute_reorder_decision
from inventory.business_constraints import (
    apply_business_constraints,
    optimize_across_budget,
    SupplierConstraints,
    BudgetConstraints,
    get_default_supplier_constraints
)
from uncertainty.dynamic_sigma import compute_rolling_forecast_error, compute_dynamic_safety_stock
from uncertainty.prediction_intervals import (
    compute_residual_statistics,
    compute_prediction_intervals,
    uncertainty_aware_reorder_decision,
    compute_forecast_uncertainty_metrics
)

logger = logging.getLogger(__name__)

class IntelligentInventoryService:
    """
    Advanced inventory service that combines:
    - Adaptive forecasting by SKU type
    - Dynamic safety stock
    - Uncertainty-aware decisions
    - Full explainability
    """
    
    def __init__(
        self,
        model=None,
        error_window_days: int = 30,
        model_feature_columns: Optional[List[str]] = None,
    ):
        self.model = model
        self.error_window_days = error_window_days
        self.historical_errors = {}
        self.model_feature_columns = model_feature_columns
        
    def get_intelligent_reorder_decision(
        self,
        sku: str,
        current_stock: int,
        demand_history: pd.Series,
        lead_time_days: int = 7,
        service_level: float = 0.95,
        last_features: Optional[pd.DataFrame] = None,
        forecast_history: Optional[pd.Series] = None,
        risk_appetite: str = "moderate",
        supplier_constraints: Optional[SupplierConstraints] = None,
        routing_service=None,
        uncertainty_service: Any | None = None,
        policy_source: str | None = None,
    ) -> Dict:
        """
        Get comprehensive reorder decision with adaptive forecasting, dynamic safety stock, uncertainty awareness, and business constraints.
        
        Parameters:
        - sku: SKU identifier
        - current_stock: Current inventory level
        - demand_history: Historical demand data
        - lead_time_days: Supplier lead time
        - service_level: Target service level
        - last_features: Recent features for ML model
        - forecast_history: Historical forecast values for error calculation
        - risk_appetite: 'conservative', 'moderate', or 'aggressive'
        - supplier_constraints: Business constraints (MOQ, order multiples, etc.)
        
        Returns:
        - Complete decision dictionary with explainability, uncertainty quantification, and business constraints
        """
        
        # 1. Classify SKU and get metadata
        sku_metadata = get_sku_classification_metadata(demand_history)
        demand_pattern = sku_metadata["demand_pattern"]
        
        # 2. Adaptive forecasting (feature schema flows through so the trained
        # model can drive inference even when the caller didn't pre-compute
        # ``last_features``).
        forecast, forecast_method, routing = adaptive_forecast(
            sku=sku,
            demand_series=demand_history,
            horizon=lead_time_days,
            model=self.model,
            last_features=last_features,
            feature_columns=self.model_feature_columns,
            routing_service=routing_service,
            include_routing=True,
        )
        
        # 3. Compute prediction intervals if forecast history available
        prediction_intervals = None
        residual_stats = None
        
        if forecast_history is not None and len(forecast_history) > 0:
            try:
                # Compute residuals
                residuals = demand_history.tail(len(forecast_history)) - forecast_history
                
                # Compute residual statistics
                residual_stats = compute_residual_statistics(residuals)
                
                # Compute prediction intervals for lead time demand
                lead_time_demand = sum(forecast[:lead_time_days])
                prediction_intervals = compute_prediction_intervals(
                    point_forecast=lead_time_demand,
                    residual_stats=residual_stats,
                    confidence_levels=[0.80, 0.90, 0.95],
                    method="normal"
                )
                
            except Exception as e:
                logger.warning("Could not compute prediction intervals for %s: %s", sku, e)
        
        # 4. Dynamic safety stock calculation
        dynamic_safety_stock = None
        sigma_fallback = demand_history.std()
        uncertainty_metadata = {
            "source": "historical_demand_std",
            "sigma": round(float(sigma_fallback) if pd.notna(sigma_fallback) else 0.0, 6),
            "sample_count": 0,
            "lookback_days": None,
            "fallback_used": True,
            "method": forecast_method,
            "horizon_days": lead_time_days,
            "demand_pattern": demand_pattern,
        }
        
        if forecast_history is not None and len(forecast_history) > 0:
            try:
                # Compute rolling forecast error
                rolling_sigma, rolling_mean_error = compute_rolling_forecast_error(
                    actuals=demand_history.tail(len(forecast_history)),
                    predictions=forecast_history,
                    window_days=self.error_window_days
                )
                
                # Compute dynamic safety stock
                dynamic_safety_stock = compute_dynamic_safety_stock(
                    rolling_sigma=rolling_sigma,
                    lead_time_days=lead_time_days,
                    service_level=service_level
                )
                
                sigma_fallback = rolling_sigma
                uncertainty_metadata.update(
                    {
                        "source": "rolling_forecast_residuals",
                        "sigma": round(float(rolling_sigma), 6),
                        "sample_count": int(len(forecast_history)),
                        "fallback_used": False,
                    }
                )

            except Exception as e:
                logger.warning("Could not compute dynamic safety stock for %s: %s", sku, e)

        if dynamic_safety_stock is None and uncertainty_service is not None:
            try:
                estimate = uncertainty_service.select_sigma(
                    sku_code=sku,
                    forecast_method=forecast_method,
                    demand_pattern=demand_pattern,
                    horizon_days=lead_time_days,
                    historical_sigma=float(sigma_fallback) if pd.notna(sigma_fallback) else 0.0,
                )
                sigma_fallback = estimate.sigma
                uncertainty_metadata = estimate.as_dict()
            except Exception as e:
                logger.warning("Residual uncertainty unavailable for %s: %s", sku, e)
        
        # 5. Compute base reorder decision
        base_decision = compute_reorder_decision(
            sku=sku,
            current_stock=current_stock,
            forecast=forecast,
            sigma=sigma_fallback,
            lead_time_days=lead_time_days,
            service_level=service_level,
            dynamic_safety_stock=dynamic_safety_stock
        )
        
        # 6. Apply uncertainty adjustments if intervals available
        if prediction_intervals and risk_appetite != "moderate":
            final_decision = uncertainty_aware_reorder_decision(
                base_decision=base_decision,
                prediction_intervals=prediction_intervals,
                risk_appetite=risk_appetite
            )
        else:
            final_decision = base_decision
        
        # 7. Apply business constraints
        if supplier_constraints is None:
            # Use default constraints based on demand pattern
            defaults = get_default_supplier_constraints()
            supplier_constraints = defaults.get(demand_pattern, SupplierConstraints())
            resolved_policy_source = "pattern_default"
        else:
            resolved_policy_source = policy_source or "sku_policy"
        
        constrained_decision = apply_business_constraints(final_decision, supplier_constraints)
        constrained_decision.setdefault("business_constraints", {})
        constrained_decision["business_constraints"]["policy_source"] = resolved_policy_source
        constrained_decision["uncertainty"] = uncertainty_metadata
        
        # 8. Add intelligence metadata
        constrained_decision.update({
            "intelligence": {
                "demand_pattern": demand_pattern,
                "forecast_method": forecast_method,
                "routing": routing,
                "sku_metadata": sku_metadata,
                "adaptive_logic": {
                    "regular_skus": "Use ML LightGBM forecast",
                    "intermittent_skus": "Use Croston's method", 
                    "highly_intermittent_skus": "Use conservative forecast with 50% buffer"
                },
                "safety_stock_approach": "dynamic" if dynamic_safety_stock is not None else "traditional",
                "uncertainty_source": uncertainty_metadata.get("source"),
                "uncertainty_quantification": prediction_intervals is not None,
                "risk_appetite": risk_appetite,
                "business_constraints_applied": True
            }
        })
        
        # 9. Add prediction intervals if available
        if prediction_intervals:
            constrained_decision["prediction_intervals"] = prediction_intervals
            constrained_decision["residual_statistics"] = residual_stats

        # 10. Expose the per-day forecast so callers can render it without
        # re-deriving from ``lead_time_demand``.
        constrained_decision["forecast_daily"] = [round(float(v), 2) for v in forecast]

        return constrained_decision
    
    def batch_reorder_decisions(
        self,
        sku_data: Dict[str, Dict],
        lead_time_days: int = 7,
        service_level: float = 0.95,
        budget_constraints: Optional[BudgetConstraints] = None
    ) -> Dict[str, Dict]:
        """
        Generate batch reorder decisions for multiple SKUs with optional budget optimization.
        
        Parameters:
        - sku_data: Dictionary with SKU as key and metadata as value
        - lead_time_days: Supplier lead time
        - service_level: Target service level
        - budget_constraints: Optional budget constraints for optimization
        
        Returns:
        - Dictionary of decisions by SKU
        """
        
        decisions = {}
        supplier_constraints_by_sku: Dict[str, SupplierConstraints] = {}

        for sku, data in sku_data.items():
            try:
                sc = data.get("supplier_constraints")
                if sc is not None:
                    supplier_constraints_by_sku[sku] = sc
                decision = self.get_intelligent_reorder_decision(
                    sku=sku,
                    current_stock=data["current_stock"],
                    demand_history=data["demand_history"],
                    lead_time_days=lead_time_days,
                    service_level=service_level,
                    last_features=data.get("last_features"),
                    forecast_history=data.get("forecast_history"),
                    supplier_constraints=sc,
                )
                decisions[sku] = decision
            except Exception as e:
                logger.warning("Error processing SKU %s: %s", sku, e)
                decisions[sku] = {"error": str(e)}

        if budget_constraints is not None:
            decisions = optimize_across_budget(
                decisions=decisions,
                budget_constraints=budget_constraints,
                supplier_constraints=supplier_constraints_by_sku,
            )

        return decisions
    
    def get_system_performance_summary(self, decisions: Dict[str, Dict]) -> Dict:
        """
        Generate summary of system performance across all SKUs.
        
        Parameters:
        - decisions: Dictionary of reorder decisions
        
        Returns:
        - Performance summary
        """
        
        total_skus = len(decisions)
        reorder_skus = sum(1 for d in decisions.values() if d.get("action") == "REORDER")
        
        # Count by demand pattern
        pattern_counts = {}
        for decision in decisions.values():
            if "intelligence" in decision:
                pattern = decision["intelligence"]["demand_pattern"]
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Safety stock methods
        safety_methods = {}
        for decision in decisions.values():
            method = decision.get("safety_stock_method", "unknown")
            safety_methods[method] = safety_methods.get(method, 0) + 1
        
        return {
            "total_skus_processed": total_skus,
            "skus_requiring_reorder": reorder_skus,
            "reorder_rate": round(reorder_skus / total_skus, 3) if total_skus > 0 else 0,
            "demand_pattern_distribution": pattern_counts,
            "safety_stock_method_distribution": safety_methods,
            "intelligent_features": {
                "adaptive_forecasting": True,
                "dynamic_safety_stock": True,
                "sku_classification": True,
                "explainability": True
            }
        }
