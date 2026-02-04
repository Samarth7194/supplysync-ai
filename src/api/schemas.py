# src/api/schemas.py

from pydantic import BaseModel
from typing import List

# ---------- Forecast ----------

class ForecastRequest(BaseModel):
    sku: str
    horizon: int = 7

class ForecastResponse(BaseModel):
    sku: str
    forecast: List[float]

# ---------- Reorder ----------

class ReorderRequest(BaseModel):
    sku: str
    current_stock: int
    lead_time_days: int = 7

class ReorderResponse(BaseModel):
    sku: str
    action: str
    order_quantity: int
    reorder_point: float
    safety_stock: float

# ---------- KPIs ----------

class KPIResponse(BaseModel):
    sku: str
    total_cost: float
    fill_rate: float
    cost_savings_pct: float
