"""Repository for analysis runs and prediction logs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import AnalysisRun, InventoryPolicy, ModelArtifact, PredictionLog, Sku


class AnalysisRepository:
    """Persist and read inventory analysis decisions."""

    def __init__(self, session: Session):
        self.session = session

    def create_analysis_run(self, values: Mapping[str, Any]) -> AnalysisRun:
        analysis = AnalysisRun(**dict(values))
        self.session.add(analysis)
        self.session.flush()
        return analysis

    def create_prediction_log(self, values: Mapping[str, Any]) -> PredictionLog:
        prediction = PredictionLog(**dict(values))
        self.session.add(prediction)
        self.session.flush()
        return prediction

    def get_sku_id(self, sku_code: str) -> int | None:
        stmt = select(Sku.id).where(Sku.sku_code == sku_code)
        return self.session.scalar(stmt)

    def active_inventory_policy_for_sku(self, sku_code: str) -> InventoryPolicy | None:
        """Return the most recent active SKU policy, if one has been configured."""
        stmt = (
            select(InventoryPolicy)
            .join(Sku)
            .where(Sku.sku_code == sku_code)
            .where(Sku.is_active.is_(True))
            .where(InventoryPolicy.is_active.is_(True))
            .order_by(InventoryPolicy.effective_from.desc(), InventoryPolicy.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def get_or_create_model_artifact(self, values: Mapping[str, Any]) -> ModelArtifact:
        model_name = str(values["model_name"])
        version = str(values["version"])
        stmt = select(ModelArtifact).where(
            ModelArtifact.model_name == model_name,
            ModelArtifact.version == version,
        )
        artifact = self.session.scalar(stmt)
        if artifact is not None:
            for key, value in dict(values).items():
                if value is not None and hasattr(artifact, key) and getattr(artifact, key) in (None, {}, []):
                    setattr(artifact, key, value)
            self.session.flush()
            return artifact

        artifact = ModelArtifact(**dict(values))
        self.session.add(artifact)
        self.session.flush()
        return artifact

    def create_analysis_with_prediction(
        self,
        analysis_values: Mapping[str, Any],
        prediction_values: Mapping[str, Any],
    ) -> tuple[AnalysisRun, PredictionLog]:
        analysis = self.create_analysis_run(analysis_values)
        payload = dict(prediction_values)
        payload.setdefault("analysis_run_id", analysis.id)
        prediction = self.create_prediction_log(payload)
        return analysis, prediction

    def recent(self, limit: int = 20) -> list[AnalysisRun]:
        limit = max(1, min(int(limit), 200))
        stmt = select(AnalysisRun).order_by(AnalysisRun.created_at.desc(), AnalysisRun.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def recent_for_sku(self, sku_code: str, limit: int = 20) -> list[AnalysisRun]:
        limit = max(1, min(int(limit), 200))
        stmt = (
            select(AnalysisRun)
            .where(AnalysisRun.sku_code == sku_code)
            .order_by(AnalysisRun.created_at.desc(), AnalysisRun.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def count(self) -> int:
        return int(self.session.scalar(select(func.count(AnalysisRun.id))) or 0)


def serialize_analysis_runs(rows: Iterable[AnalysisRun]) -> list[dict[str, Any]]:
    """Serialize SQLAlchemy analysis rows to the legacy recent-analyses shape."""
    items: list[dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "id": row.id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "sku": row.sku_code,
                "risk": row.risk,
                "action": row.action,
                "current_stock": float(row.current_stock) if row.current_stock is not None else None,
                "recommended_order": row.recommended_order_quantity,
                "demand_pattern": row.demand_pattern,
                "forecast_method": row.forecast_method,
                "demand_source": row.demand_source,
                "forecast_source": row.forecast_source,
                "model_type": None,
                "model_name": None,
                "artifact_available": None,
                "lead_time_demand": float(row.lead_time_demand) if row.lead_time_demand is not None else None,
                "safety_stock": float(row.safety_stock) if row.safety_stock is not None else None,
                "reorder_point": float(row.reorder_point) if row.reorder_point is not None else None,
                "inventory_gap": float(row.inventory_gap) if row.inventory_gap is not None else None,
                "p50": float(row.p50) if row.p50 is not None else None,
                "p90": float(row.p90) if row.p90 is not None else None,
            }
        )
    return items
