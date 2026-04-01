# src/services/explainability_service.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

class ExplainabilityService:
    """
    Comprehensive explainability service for inventory decisions.
    Makes every recommendation transparent and understandable.
    """
    
    def generate_decision_explanation(self, decision: Dict) -> Dict:
        """
        Generate human-readable explanation for a reorder decision.
        
        Parameters:
        - decision: Reorder decision dictionary
        
        Returns:
        - Structured explanation
        """
        
        sku = decision.get("sku", "Unknown")
        action = decision.get("action", "NO_ACTION")
        
        explanation = {
            "summary": self._generate_summary(decision),
            "key_factors": self._extract_key_factors(decision),
            "calculation_breakdown": self._generate_calculation_breakdown(decision),
            "risk_assessment": self._generate_risk_assessment(decision),
            "business_impact": self._generate_business_impact(decision),
            "recommendations": self._generate_recommendations(decision),
            "metadata": {
                "sku": sku,
                "timestamp": datetime.now().isoformat(),
                "decision_action": action
            }
        }
        
        return explanation
    
    def _generate_summary(self, decision: Dict) -> str:
        """Generate executive summary of the decision."""
        
        sku = decision.get("sku", "Unknown")
        action = decision.get("action", "NO_ACTION")
        order_qty = decision.get("order_quantity", 0)
        current_stock = decision.get("current_stock", 0)
        
        if action == "REORDER":
            return f"Order {order_qty:,} units of SKU {sku}. Current stock ({current_stock:,}) is below reorder point, requiring immediate action to maintain service levels."
        else:
            return f"No order needed for SKU {sku}. Current stock ({current_stock:,}) is sufficient to meet expected demand during lead time."
    
    def _extract_key_factors(self, decision: Dict) -> List[Dict]:
        """Extract and rank key factors influencing the decision."""
        
        factors = []
        
        # Current stock position
        current_stock = decision.get("current_stock", 0)
        reorder_point = decision.get("reorder_point", 0)
        stock_ratio = current_stock / reorder_point if reorder_point > 0 else 1
        
        factors.append({
            "factor": "Current Inventory Position",
            "value": f"{current_stock:,} units",
            "impact": "high" if stock_ratio < 0.8 else "medium",
            "description": f"Current stock is {stock_ratio:.1%} of reorder point"
        })
        
        # Lead time demand
        lead_time_demand = decision.get("lead_time_demand", 0)
        factors.append({
            "factor": "Expected Lead Time Demand",
            "value": f"{lead_time_demand:,.0f} units",
            "impact": "high",
            "description": f"Expected demand during {decision.get('lead_time_days', 7)}-day supplier lead time"
        })
        
        # Safety stock
        safety_stock = decision.get("safety_stock", 0)
        safety_method = decision.get("safety_stock_method", "traditional")
        factors.append({
            "factor": "Safety Stock Buffer",
            "value": f"{safety_stock:,.0f} units",
            "impact": "medium",
            "description": f"Buffer calculated using {safety_method} method for {decision.get('service_level', 0.95):.0%} service level"
        })
        
        # Demand pattern
        intelligence = decision.get("intelligence", {})
        demand_pattern = intelligence.get("demand_pattern", "unknown")
        forecast_method = intelligence.get("forecast_method", "unknown")
        
        factors.append({
            "factor": "Demand Pattern & Forecast Method",
            "value": f"{demand_pattern.title()} demand using {forecast_method}",
            "impact": "medium",
            "description": f"SKU classified as {demand_pattern} demand, using {forecast_method} forecasting approach"
        })
        
        # Business constraints
        business_constraints = decision.get("business_constraints")
        if business_constraints:
            constraints_applied = business_constraints.get("constraints_applied", [])
            if constraints_applied:
                factors.append({
                    "factor": "Business Constraints Applied",
                    "value": f"{len(constraints_applied)} adjustments",
                    "impact": "medium",
                    "description": f"Applied supplier constraints: {', '.join(constraints_applied)}"
                })
        
        # Uncertainty adjustments
        uncertainty_adj = decision.get("uncertainty_adjustment")
        if uncertainty_adj:
            risk_appetite = uncertainty_adj.get("risk_appetite", "moderate")
            factors.append({
                "factor": "Risk Appetite Adjustment",
                "value": f"{risk_appetite.title()} approach",
                "impact": "medium",
                "description": f"Decision adjusted for {risk_appetite} risk appetite"
            })
        
        return factors
    
    def _generate_calculation_breakdown(self, decision: Dict) -> Dict:
        """Generate step-by-step calculation breakdown."""
        
        explainability = decision.get("explainability", {})
        
        breakdown = {
            "step_1_lead_time_demand": {
                "title": "Step 1: Calculate Lead Time Demand",
                "formula": explainability.get("lead_time_demand_calculation", "Sum of forecast for lead time days"),
                "inputs": {
                    "forecast_days": decision.get("lead_time_days", 7),
                    "daily_forecast_avg": decision.get("lead_time_demand", 0) / decision.get("lead_time_days", 1)
                },
                "result": f"{decision.get('lead_time_demand', 0):,.0f} units"
            },
            
            "step_2_safety_stock": {
                "title": "Step 2: Calculate Safety Stock",
                "formula": explainability.get("safety_stock_calculation", "Z × σ × √LeadTime"),
                "inputs": {
                    "service_level": f"{decision.get('service_level', 0.95):.0%}",
                    "z_score": decision.get("z_score", 1.65),
                    "forecast_error_std": "σ (estimated from historical errors)",
                    "lead_time_days": decision.get("lead_time_days", 7)
                },
                "result": f"{decision.get('safety_stock', 0):,.0f} units"
            },
            
            "step_3_reorder_point": {
                "title": "Step 3: Calculate Reorder Point",
                "formula": explainability.get("reorder_point_calculation", "Lead Time Demand + Safety Stock"),
                "inputs": {
                    "lead_time_demand": decision.get("lead_time_demand", 0),
                    "safety_stock": decision.get("safety_stock", 0)
                },
                "result": f"{decision.get('reorder_point', 0):,.0f} units"
            },
            
            "step_4_order_quantity": {
                "title": "Step 4: Calculate Order Quantity",
                "formula": explainability.get("decision_logic", "Order if Current Stock < Reorder Point"),
                "inputs": {
                    "reorder_point": decision.get("reorder_point", 0),
                    "current_stock": decision.get("current_stock", 0),
                    "comparison": f"{decision.get('current_stock', 0)} < {decision.get('reorder_point', 0)} = {decision.get('current_stock', 0) < decision.get('reorder_point', 0)}"
                },
                "result": f"{decision.get('order_quantity', 0):,.0f} units"
            }
        }
        
        return breakdown
    
    def _generate_risk_assessment(self, decision: Dict) -> Dict:
        """Generate risk assessment for the decision."""
        
        current_stock = decision.get("current_stock", 0)
        reorder_point = decision.get("reorder_point", 0)
        safety_stock = decision.get("safety_stock", 0)
        
        # Calculate risk metrics
        stock_coverage_ratio = current_stock / reorder_point if reorder_point > 0 else 1
        safety_stock_ratio = safety_stock / reorder_point if reorder_point > 0 else 0
        
        # Determine risk level
        if stock_coverage_ratio < 0.5:
            risk_level = "high"
        elif stock_coverage_ratio < 0.8:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        assessment = {
            "overall_risk_level": risk_level,
            "stock_coverage_ratio": round(stock_coverage_ratio, 2),
            "safety_stock_ratio": round(safety_stock_ratio, 2),
            "risk_factors": [],
            "mitigation_strategies": []
        }
        
        # Identify risk factors
        if stock_coverage_ratio < 0.5:
            assessment["risk_factors"].append("Very low inventory coverage - high stockout risk")
        
        if safety_stock_ratio < 0.2:
            assessment["risk_factors"].append("Low safety stock buffer - vulnerable to demand spikes")
        
        intelligence = decision.get("intelligence", {})
        demand_pattern = intelligence.get("demand_pattern", "")
        if demand_pattern == "highly_intermittent":
            assessment["risk_factors"].append("Highly intermittent demand - forecast uncertainty")
        
        # Suggest mitigation strategies
        if risk_level == "high":
            assessment["mitigation_strategies"].append("Expedite order if possible")
            assessment["mitigation_strategies"].append("Consider alternative suppliers")
        
        if safety_stock_ratio < 0.2:
            assessment["mitigation_strategies"].append("Increase safety stock buffer")
            assessment["mitigation_strategies"].append("Improve forecast accuracy")
        
        return assessment
    
    def _generate_business_impact(self, decision: Dict) -> Dict:
        """Generate business impact analysis."""
        
        action = decision.get("action", "NO_ACTION")
        order_qty = decision.get("order_quantity", 0)
        
        impact = {
            "financial_impact": {},
            "operational_impact": {},
            "service_impact": {}
        }
        
        if action == "REORDER":
            # Estimate financial impact (simplified)
            estimated_cost_per_unit = 10.0  # Would come from product catalog
            total_order_value = order_qty * estimated_cost_per_unit
            
            impact["financial_impact"] = {
                "order_value": f"${total_order_value:,.0f}",
                "carrying_cost_increase": f"${total_order_value * 0.25:,.0f} annually",  # 25% carrying cost
                "stockout_cost_avoided": "Potential savings from preventing stockouts"
            }
            
            impact["operational_impact"] = {
                "storage_space_required": f"{order_qty * 0.1:,.0f} cubic feet",  # Estimate
                "handling_complexity": "Medium" if order_qty > 100 else "Low",
                "supplier_coordination": "Required"
            }
            
            impact["service_impact"] = {
                "service_level_maintained": f"{decision.get('service_level', 0.95):.0%}",
                "stockout_risk_reduction": "Significant",
                "customer_satisfaction": "Protected"
            }
        
        else:  # NO_ACTION
            impact["financial_impact"] = {
                "immediate_cost_savings": "No ordering cost",
                "carrying_cost_avoided": "Reduced inventory holding costs",
                "opportunity_cost": "Potential stockout cost if demand spikes"
            }
            
            impact["operational_impact"] = {
                "storage_space_saved": "No additional space required",
                "administrative_effort": "Minimal",
                "supplier_coordination": "Not required"
            }
            
            current_stock = decision.get("current_stock", 0)
            reorder_point = decision.get("reorder_point", 0)
            coverage_ratio = current_stock / reorder_point if reorder_point > 0 else 1
            
            impact["service_impact"] = {
                "service_level_risk": "Low" if coverage_ratio > 1.2 else "Medium",
                "stockout_probability": f"{max(0, (1 - coverage_ratio) * 100):.0f}%",
                "customer_satisfaction": "Maintained"
            }
        
        return impact
    
    def _generate_recommendations(self, decision: Dict) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        action = decision.get("action", "NO_ACTION")
        
        if action == "REORDER":
            order_qty = decision.get("order_quantity", 0)
            recommendations.append(f"Place order for {order_qty:,} units immediately to maintain service levels")
            
            # Additional recommendations based on context
            intelligence = decision.get("intelligence", {})
            demand_pattern = intelligence.get("demand_pattern", "")
            
            if demand_pattern == "intermittent":
                recommendations.append("Monitor demand closely - intermittent patterns may require manual review")
            
            if decision.get("safety_stock_method") == "dynamic":
                recommendations.append("Track forecast accuracy to validate dynamic safety stock adjustments")
            
            business_constraints = decision.get("business_constraints")
            if business_constraints:
                constraints_applied = business_constraints.get("constraints_applied", [])
                if any("MOQ" in str(c) for c in constraints_applied):
                    recommendations.append("Consider negotiating MOQ terms with supplier for better flexibility")
        
        else:  # NO_ACTION
            current_stock = decision.get("current_stock", 0)
            reorder_point = decision.get("reorder_point", 0)
            
            recommendations.append(f"No immediate action required - current stock of {current_stock:,} units is sufficient")
            
            if current_stock > reorder_point * 1.5:
                recommendations.append("Consider inventory optimization - stock levels may be excessive")
            
            recommendations.append(f"Next review recommended when stock falls below {reorder_point * 1.2:,.0f} units")
        
        # General recommendations
        recommendations.append("Monitor actual demand vs forecast to improve accuracy")
        recommendations.append("Review supplier lead times for potential optimization")
        
        return recommendations
    
    def generate_batch_explanation(self, decisions: Dict[str, Dict]) -> Dict:
        """
        Generate explanation for batch of decisions.
        
        Parameters:
        - decisions: Dictionary of decisions by SKU
        
        Returns:
        - Batch-level explanation
        """
        
        total_skus = len(decisions)
        reorder_skus = sum(1 for d in decisions.values() if d.get("action") == "REORDER")
        
        # Calculate totals
        total_order_value = 0
        total_safety_stock = 0
        
        for decision in decisions.values():
            if decision.get("action") == "REORDER":
                order_qty = decision.get("order_quantity", 0)
                # Estimate value (would use actual cost in production)
                total_order_value += order_qty * 10.0
            
            total_safety_stock += decision.get("safety_stock", 0)
        
        batch_explanation = {
            "executive_summary": f"Review of {total_skus} SKUs recommends reordering {reorder_skus} items with estimated value of ${total_order_value:,.0f}",
            "key_metrics": {
                "total_skus_reviewed": total_skus,
                "skus_requiring_reorder": reorder_skus,
                "reorder_rate": f"{(reorder_skus/total_skus)*100:.1f}%",
                "total_order_quantity": sum(d.get("order_quantity", 0) for d in decisions.values()),
                "estimated_order_value": f"${total_order_value:,.0f}",
                "total_safety_stock_buffer": f"{total_safety_stock:,.0f} units"
            },
            "risk_distribution": self._analyze_batch_risks(decisions),
            "action_items": self._generate_batch_actions(decisions),
            "individual_explanations": {
                sku: self.generate_decision_explanation(decision)
                for sku, decision in decisions.items()
            }
        }
        
        return batch_explanation
    
    def _analyze_batch_risks(self, decisions: Dict[str, Dict]) -> Dict:
        """Analyze risk distribution across batch."""
        
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        
        for decision in decisions.values():
            current_stock = decision.get("current_stock", 0)
            reorder_point = decision.get("reorder_point", 0)
            
            if reorder_point > 0:
                coverage_ratio = current_stock / reorder_point
                
                if coverage_ratio < 0.5:
                    high_risk += 1
                elif coverage_ratio < 0.8:
                    medium_risk += 1
                else:
                    low_risk += 1
        
        return {
            "high_risk_skus": high_risk,
            "medium_risk_skus": medium_risk,
            "low_risk_skus": low_risk,
            "risk_percentage": {
                "high": f"{(high_risk/len(decisions))*100:.1f}%",
                "medium": f"{(medium_risk/len(decisions))*100:.1f}%",
                "low": f"{(low_risk/len(decisions))*100:.1f}%"
            }
        }
    
    def _generate_batch_actions(self, decisions: Dict[str, Dict]) -> List[str]:
        """Generate batch-level action items."""
        
        actions = []
        
        # Count reorder actions
        reorder_count = sum(1 for d in decisions.values() if d.get("action") == "REORDER")
        
        if reorder_count > 0:
            actions.append(f"Process {reorder_count} purchase orders immediately")
            actions.append("Review total order value against budget constraints")
        
        # Check for high-risk items
        high_risk_items = [
            sku for sku, decision in decisions.items()
            if decision.get("current_stock", 0) < decision.get("reorder_point", 0) * 0.5
        ]
        
        if high_risk_items:
            actions.append(f"Urgent review required for {len(high_risk_items)} high-risk SKUs: {', '.join(high_risk_items[:3])}{'...' if len(high_risk_items) > 3 else ''}")
        
        # General actions
        actions.append("Update inventory records after orders are placed")
        actions.append("Schedule follow-up review for low-risk items")
        actions.append("Analyze demand patterns to improve forecasting accuracy")
        
        return actions
