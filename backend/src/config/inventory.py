"""Inventory-planning defaults and validation."""

from __future__ import annotations

from dataclasses import dataclass

from config.common import env_float, env_int


@dataclass(frozen=True)
class InventorySettings:
    default_lead_time_days: int
    default_service_level: float
    holding_cost_per_unit: float
    stockout_cost_per_unit: float


def load_inventory_settings() -> InventorySettings:
    lead_time_days = env_int("LEAD_TIME_DAYS", 7)
    service_level = env_float("SERVICE_LEVEL", 0.95)
    holding_cost = env_float("HOLDING_COST", 0.2)
    stockout_cost = env_float("STOCKOUT_COST", 1.0)

    if lead_time_days <= 0:
        raise ValueError("LEAD_TIME_DAYS must be greater than 0.")
    if not 0 < service_level < 1:
        raise ValueError("SERVICE_LEVEL must satisfy 0 < SERVICE_LEVEL < 1.")
    if holding_cost < 0:
        raise ValueError("HOLDING_COST must be greater than or equal to 0.")
    if stockout_cost < 0:
        raise ValueError("STOCKOUT_COST must be greater than or equal to 0.")

    return InventorySettings(
        default_lead_time_days=lead_time_days,
        default_service_level=service_level,
        holding_cost_per_unit=holding_cost,
        stockout_cost_per_unit=stockout_cost,
    )

