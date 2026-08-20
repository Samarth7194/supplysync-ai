# backend/main.py

import hmac
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, List, Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure 'src' is accessible for intra-package imports like `from services...`
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from services.data_service import DataService
from services.intelligent_inventory_service import IntelligentInventoryService
from services.model_service import get_model_service
from services.model_service import ModelArtifactValidationError
from ingestion.load_retail_data import get_sku_descriptions
from config.settings import load_settings
from dependencies.analysis import get_analysis_repository
from dependencies.stock import get_stock_service
from db.session import database_health
from db.session import get_session
from repositories.analysis_repository import AnalysisRepository
from repositories.model_monitoring_repository import ModelMonitoringRepository
from repositories.retraining_repository import RetrainingRepository
from services.analysis_service import (
    AnalysisExecutionError,
    AnalysisService,
    AnalysisServiceUnavailableError,
)
from services.model_monitoring_service import ModelMonitoringService
from services.retraining_decision_service import RetrainingDecisionService
from services.stock_service import (
    InvalidStockLevelError,
    StockPersistenceUnavailableError,
    StockService,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from auth.session import (
    AuthConfig,
    COOKIE_NAME,
    check_credentials,
    sign_session,
    verify_session,
)

logger = logging.getLogger("supplysync")

# --- App State ---

_loaded_model = None
_data_service: Optional[DataService] = None
_sku_descriptions: dict = {}
_inventory_service: Optional[IntelligentInventoryService] = None
_model_artifact_status: dict = {}

# --- Runtime configuration --------------------------------------------------

SETTINGS = load_settings()


# --- Access control ---------------------------------------------------------
#
# Two-tier, intentionally minimal:
#   1. AUTH_MODE=demo  — require a valid session cookie OR a valid X-API-Key.
#   2. AUTH_MODE=off (default) — endpoints remain open; the X-API-Key header
#      still works if ``API_KEY`` is set, preserving the original behavior.
#
# ``require_access`` is the single dependency gate every ``/api/*`` route
# uses. ``/health`` and ``/api/auth/*`` deliberately stay public.

AUTH_CONFIG: AuthConfig = SETTINGS.auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_access(
    api_key: Optional[str] = Security(api_key_header),
    session_cookie: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    # Legacy header auth — valid whenever API_KEY is configured, regardless
    # of AUTH_MODE, so automation scripts can keep working.
    if AUTH_CONFIG.api_key and api_key and hmac.compare_digest(api_key, AUTH_CONFIG.api_key):
        return {"source": "api_key"}

    if AUTH_CONFIG.enabled:
        payload = verify_session(session_cookie or "", AUTH_CONFIG)
        if payload:
            return {"source": "session", "user": payload.get("sub")}
        raise HTTPException(
            status_code=401,
            detail="Authentication required. POST /api/auth/login with the demo credentials.",
        )

    # AUTH_MODE=off and no legacy key configured → open access.
    if not AUTH_CONFIG.api_key:
        return {"source": "anonymous"}

    # API_KEY is configured but header missing/wrong.
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# Backward-compatible alias so existing route signatures keep working.
verify_api_key = require_access


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded_model, _data_service, _sku_descriptions, _inventory_service, _model_artifact_status

    model_service = get_model_service(model_dir=SETTINGS.forecasting.model_path)
    model_feature_columns: Optional[List[str]] = None
    try:
        _model_artifact_status = model_service.artifact_status("lightgbm_demand_forecast")
        _loaded_model = model_service.load_model("lightgbm_demand_forecast")
        metadata = model_service.get_model_metadata("lightgbm_demand_forecast")
        model_feature_columns = metadata.get("features") if metadata else None
        logger.info(
            "Loaded forecasting model; feature schema has %d columns",
            len(model_feature_columns) if model_feature_columns else 0,
        )
    except FileNotFoundError:
        _loaded_model = None
        logger.warning("Trained forecasting model not found; inference will fall back")
    except ModelArtifactValidationError as exc:
        _loaded_model = None
        _model_artifact_status = model_service.artifact_status("lightgbm_demand_forecast")
        logger.warning(
            "Trained forecasting model failed validation; inference will fall back",
            extra={"operation": "load_model", "model_name": "lightgbm_demand_forecast", "exception_type": type(exc).__name__},
        )

    try:
        _data_service = DataService.get_instance()
    except Exception as e:
        logger.warning("DataService unavailable: %s", e)
        _data_service = None

    try:
        _sku_descriptions = get_sku_descriptions()
    except Exception as e:
        logger.warning("SKU descriptions unavailable: %s", e)
        _sku_descriptions = {}

    _inventory_service = IntelligentInventoryService(
        model=_loaded_model,
        model_feature_columns=model_feature_columns,
    )

    yield


# --- App Setup ---

app = FastAPI(title="SupplySync AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.app.allowed_origins,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "X-API-Key"],
    # Needed so the demo-mode session cookie crosses origins during dev
    # (frontend on :3000, backend on :8000). In production with a single
    # origin this is a no-op.
    allow_credentials=True,
)


def get_analysis_service(
    analysis_repository: AnalysisRepository = Depends(get_analysis_repository),
) -> AnalysisService:
    """Build the application service for analysis requests.

    The route stays thin, but the dependency remains in main.py because it
    needs access to lifespan-initialized app state such as the loaded model,
    data service, and inventory service.
    """
    return AnalysisService(
        inventory_service=_inventory_service,
        settings=SETTINGS,
        data_service=_data_service,
        analysis_repository=analysis_repository,
        model_loaded=_loaded_model is not None,
        model_dir=SETTINGS.forecasting.model_path,
    )


def get_model_monitoring_service(
    session: Session = Depends(get_session),
) -> ModelMonitoringService:
    """Build the monitoring service for API requests without touching inference."""
    return ModelMonitoringService(
        repository=ModelMonitoringRepository(session),
        settings=SETTINGS,
        data_service=_data_service,
        offline_evaluation_path=Path(__file__).resolve().parent / "data" / "forecast_evaluation.json",
    )


def get_retraining_decision_service(
    session: Session = Depends(get_session),
) -> RetrainingDecisionService:
    """Build the retraining recommendation service.

    Phase E only evaluates/persists recommendations when explicitly requested
    by scripts. This dependency is read-only for the status API.
    """
    return RetrainingDecisionService(
        repository=RetrainingRepository(session),
        settings=SETTINGS,
    )


# --- Schemas ---

class AnalyzeRequest(BaseModel):
    sku: str = Field(..., min_length=1)
    current_stock: float = Field(50, ge=0)
    demand_history: Optional[List[float]] = None
    # Configurable business assumptions. Both are optional and fall back to
    # typed runtime settings for backward compat.
    lead_time_days: Optional[int] = Field(None, ge=1, le=90)
    service_level: Optional[float] = Field(None, gt=0.5, lt=1.0)


class StockUpdateRequest(BaseModel):
    quantity_on_hand: float = Field(..., ge=0)
    quantity_reserved: float = Field(0, ge=0)
    note: Optional[str] = Field(None, max_length=500)
    sku_name: Optional[str] = Field(None, max_length=500)


class StockResponse(BaseModel):
    sku: str
    quantity_on_hand: float
    quantity_reserved: float
    quantity_available: float
    source: str
    recorded_at: Optional[str] = None


class StockListResponse(BaseModel):
    items: List[StockResponse]
    source: str = "database"


class ForecastBlock(BaseModel):
    p50: float
    p90: float
    daily: List[float]
    full_horizon_daily: List[float] = Field(default_factory=list)
    horizon_days: int = 7


class ModelInfo(BaseModel):
    """Compact model/method provenance.

    Always truthful: for statistical and rule-based paths ``model_type`` and
    ``model_name`` reflect the actual method used, not a hallucinated ML model.
    """
    model_name: str
    model_type: str                  # "ml" | "statistical_method" | "rule_based_fallback" | "none"
    artifact_available: bool         # whether a saved model file is loaded
    model_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    artifact_valid: Optional[bool] = None
    trained_at: Optional[str] = None
    feature_count: Optional[int] = None
    dataset: Optional[str] = None
    evaluation_available: bool = False
    evaluation_generated_at: Optional[str] = None


MonitoringStatus = Literal["unavailable", "insufficient_evidence", "stable", "warning", "degraded"]
PersistedMonitoringStatus = Literal["insufficient_evidence", "stable", "warning", "degraded"]


class ModelMonitoringSnapshotResponse(BaseModel):
    model_artifact_id: Optional[int] = None
    model_name: str
    model_version: Optional[str] = None
    lifecycle_status: Optional[str] = None
    generated_at: Optional[str] = None
    status: MonitoringStatus
    degradation_reason: Optional[str] = None
    degradation_message: Optional[str] = None
    evaluation_count: int = 0
    window_type: Optional[str] = None
    window_size: Optional[int] = None
    metric_wape: Optional[float] = None
    metric_mae: Optional[float] = None
    metric_rmse: Optional[float] = None
    metric_bias: Optional[float] = None
    metric_mase: Optional[float] = None
    residual_mean: Optional[float] = None
    residual_std: Optional[float] = None
    baseline_wape: Optional[float] = None
    baseline_provenance: Optional[str] = None
    wape_relative_change: Optional[float] = None
    bias_ratio: Optional[float] = None
    consecutive_degradation_count: int = 0
    created: Optional[bool] = None


class ModelMonitoringHistoryResponse(BaseModel):
    items: List[ModelMonitoringSnapshotResponse]
    limit: int
    count: int


class ModelRetrainingStatusResponse(BaseModel):
    recommended: bool
    reason: str
    message: str
    latest_monitoring_status: Optional[MonitoringStatus] = None
    new_evaluated_forecast_days: int
    minimum_required: int
    cooldown_days: int
    cooldown_remaining_days: int
    baseline_model: Optional[dict] = None
    source_monitoring_snapshot_id: Optional[int] = None
    last_retraining_attempt: Optional[str] = None
    retraining_run_id: Optional[int] = None
    automatic_execution_enabled: bool


class ExplanationBlock(BaseModel):
    """Compact, deterministic explainability for the routing + recommendation.

    Every field is template-generated from real inputs (no LLM, no fabricated
    confidence score). Reads: *why* the SKU was classified this way, *why*
    the forecasting path was chosen, *why* the risk bucket landed where it
    did, and a one-sentence confidence caveat when appropriate.
    """
    classification_reason: str
    method_reason: str
    risk_reason: str
    confidence_note: str


class DecisionBlock(BaseModel):
    """Explains why the recommendation is what it is.

    Every field is already computed by ``IntelligentInventoryService``; we
    surface them so the frontend (and any API consumer) can show the
    inventory-decision reasoning instead of just the final number.
    """
    lead_time_days: int
    lead_time_demand: float
    safety_stock: float
    safety_stock_method: str
    reorder_point: float
    service_level: float
    inventory_gap: float       # max(0, reorder_point - current_stock)
    why: str                   # human-readable one-liner
    constraints: dict = Field(default_factory=dict)
    uncertainty: dict = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    sku: str
    risk: str
    risk_color: str
    forecast: ForecastBlock
    current_stock: float
    recommended_order: int
    action: str
    demand_pattern: str
    forecast_method: str
    # Provenance fields (backward-compatible additions):
    #   demand_source   — where the input demand series came from
    #   forecast_source — how the forecast was produced
    demand_source: str
    forecast_source: str
    decision: DecisionBlock
    model_info: ModelInfo
    explanation: ExplanationBlock


# --- Routes ---

_MISSING_DATA_MSG = (
    "Data service unavailable — processed dataset missing. "
    "Generate it with: cd backend && python scripts/bootstrap.py"
)


# --- Auth endpoints ---------------------------------------------------------


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


@app.get("/api/auth/status")
async def auth_status(
    session_cookie: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Public probe so the frontend can decide whether to show a login gate.

    Tells the UI whether demo-mode auth is on, and — if so — whether the
    current browser session is already authenticated.
    """
    authenticated = False
    user: Optional[str] = None
    if AUTH_CONFIG.enabled and session_cookie:
        payload = verify_session(session_cookie, AUTH_CONFIG)
        if payload:
            authenticated = True
            user = payload.get("sub")
    return {
        "auth_mode": AUTH_CONFIG.mode,
        "authenticated": authenticated,
        "user": user,
        "login_required": AUTH_CONFIG.enabled and not authenticated,
        "note": (
            "Demo auth: a single shared credential from DEMO_USER/DEMO_PASSWORD. "
            "Not a production identity system."
            if AUTH_CONFIG.enabled else
            "Auth is off. Any caller can hit /api/* (X-API-Key still honored if configured)."
        ),
    }


@app.post("/api/auth/login")
async def auth_login(body: LoginRequest, response: Response):
    if not AUTH_CONFIG.enabled:
        # Nothing to authenticate against; be explicit rather than silently OK.
        raise HTTPException(
            status_code=400,
            detail="Auth is disabled (AUTH_MODE=off). No login required.",
        )
    if not check_credentials(body.username, body.password, AUTH_CONFIG):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = sign_session(body.username, AUTH_CONFIG)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=AUTH_CONFIG.ttl_seconds,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"ok": True, "user": body.username, "expires_in": AUTH_CONFIG.ttl_seconds}


@app.post("/api/auth/logout")
async def auth_logout(response: Response):
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me", dependencies=[Depends(require_access)])
async def auth_me(
    session_cookie: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Returns the authenticated user (or anonymous/api_key equivalent)."""
    if AUTH_CONFIG.enabled and session_cookie:
        payload = verify_session(session_cookie, AUTH_CONFIG)
        if payload:
            return {"user": payload.get("sub"), "source": "session"}
    # Reached via api_key or AUTH_MODE=off.
    return {"user": None, "source": "api_key_or_anonymous"}


@app.get("/health")
async def health():
    """Liveness + setup-readiness probe.

    ``status`` is "online" as long as the process is serving. The remaining
    fields report which backing artifacts are available so a developer (or a
    Docker healthcheck) can tell at a glance what still needs generating.
    """
    kpi_cache = Path(__file__).resolve().parent / "data" / "cached_kpis.json"
    database = database_health()
    return {
        "status": "online",
        "model_loaded": _loaded_model is not None,
        "data_available": _data_service is not None,
        "kpis_available": kpi_cache.exists(),
        "database": database,
        "model_artifact": {
            "valid": bool(_model_artifact_status.get("valid", _loaded_model is not None)),
            "model_name": _model_artifact_status.get("model_name", "lightgbm_demand_forecast"),
            "version": _model_artifact_status.get("version"),
            "feature_schema_version": _model_artifact_status.get("feature_schema_version"),
            "lifecycle_status": _model_artifact_status.get("lifecycle_status"),
        },
        "hint": None if (_loaded_model and _data_service and kpi_cache.exists())
                else "Run `python scripts/bootstrap.py` to generate missing artifacts.",
    }


@app.get("/api/analyses/recent", dependencies=[Depends(verify_api_key)])
async def recent_analyses(
    limit: int = 20,
    analysis_service: AnalysisService = Depends(get_analysis_service),
):
    """Return the most-recent persisted /api/analyze snapshots.

    SQLAlchemy-backed analysis history.
    """
    try:
        return analysis_service.recent_analyses(limit=limit)
    except AnalysisExecutionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/model-info", dependencies=[Depends(verify_api_key)])
async def model_info():
    """Describes the currently loaded forecasting artifact.

    Returned fields are consistent with the per-analyze ``model_info`` block,
    but this endpoint always reports on the trained ML artifact (not a
    specific analyze call's method). Useful for a "model status" surface in
    the UI.
    """
    backend_dir = Path(__file__).resolve().parent
    model_service = get_model_service(model_dir=str(backend_dir / "saved_models"))
    meta = model_service.get_model_metadata("lightgbm_demand_forecast") or {}
    artifact_status = model_service.artifact_status("lightgbm_demand_forecast")

    eval_path = backend_dir / "data" / "forecast_evaluation.json"
    eval_available = eval_path.exists()
    eval_generated_at: Optional[str] = None
    eval_summary: Optional[dict] = None
    if eval_available:
        try:
            with eval_path.open() as fh:
                payload = json.load(fh)
            eval_generated_at = payload.get("generated_at")
            eval_summary = (payload.get("aggregates") or {}).get("all", {}).get("lightgbm")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluation artifact unreadable: %s", exc)

    artifact_available = _loaded_model is not None
    features = meta.get("features") or []
    return {
        "model_name": "lightgbm_demand_forecast",
        "model_type": "ml",
        "artifact_available": artifact_available,
        "model_version": meta.get("version"),
        "feature_schema_version": meta.get("feature_schema_version"),
        "artifact_valid": bool(artifact_status.get("valid")),
        "lifecycle_status": meta.get("lifecycle_status"),
        "trained_at": meta.get("saved_at"),
        "dataset": meta.get("dataset"),
        "feature_count": len(features) if features else None,
        "features": features if features else None,
        "train_skus": meta.get("train_skus"),
        "training_metrics": {
            "mae": meta.get("mae"),
            "rmse": meta.get("rmse"),
            # MAPE is reported by the trainer but intentionally omitted here;
            # use the evaluation artifact for interpretable metrics.
        },
        "evaluation": {
            "available": eval_available,
            "generated_at": eval_generated_at,
            "summary": eval_summary,  # aggregate 'all' row for lightgbm, if present
        },
        "hint": None if artifact_available else (
            "Trained model not loaded. Run `python scripts/bootstrap.py` to "
            "build the artifact; until then regular-demand forecasts fall "
            "back to the 7-day moving average."
        ),
    }


def _to_float(value) -> Optional[float]:
    return float(value) if value is not None else None


def _unavailable_monitoring_response() -> ModelMonitoringSnapshotResponse:
    return ModelMonitoringSnapshotResponse(
        model_name="lightgbm_demand_forecast",
        status="unavailable",
        degradation_reason="monitoring_unavailable",
        degradation_message="No model monitoring snapshot has been created yet.",
        baseline_provenance="unavailable",
    )


def _monitoring_response(snapshot, *, created: Optional[bool] = None) -> ModelMonitoringSnapshotResponse:
    artifact = getattr(snapshot, "model_artifact", None)
    return ModelMonitoringSnapshotResponse(
        model_artifact_id=snapshot.model_artifact_id,
        model_name=snapshot.model_name,
        model_version=snapshot.model_version,
        lifecycle_status=getattr(artifact, "lifecycle_status", None),
        generated_at=snapshot.generated_at.isoformat() if snapshot.generated_at else None,
        status=snapshot.status,
        degradation_reason=snapshot.degradation_reason,
        degradation_message=snapshot.degradation_message,
        evaluation_count=int(snapshot.evaluation_count or 0),
        window_type=snapshot.window_type,
        window_size=snapshot.window_size,
        metric_wape=_to_float(snapshot.metric_wape),
        metric_mae=_to_float(snapshot.metric_mae),
        metric_rmse=_to_float(snapshot.metric_rmse),
        metric_bias=_to_float(snapshot.metric_bias),
        metric_mase=_to_float(snapshot.metric_mase),
        residual_mean=_to_float(snapshot.residual_mean),
        residual_std=_to_float(snapshot.residual_std),
        baseline_wape=_to_float(snapshot.baseline_wape),
        baseline_provenance=snapshot.baseline_provenance,
        wape_relative_change=_to_float(snapshot.wape_relative_change),
        bias_ratio=_to_float(snapshot.bias_ratio),
        consecutive_degradation_count=int(snapshot.consecutive_degradation_count or 0),
        created=created,
    )


def _retraining_status_response(decision) -> ModelRetrainingStatusResponse:
    baseline_model = None
    if decision.baseline_model_artifact_id is not None:
        baseline_model = {
            "artifact_id": decision.baseline_model_artifact_id,
            "model_name": decision.baseline_model_name,
            "model_version": decision.baseline_model_version,
        }
    return ModelRetrainingStatusResponse(
        recommended=decision.recommended,
        reason=decision.reason,
        message=decision.message,
        latest_monitoring_status=decision.latest_monitoring_status,
        new_evaluated_forecast_days=decision.new_evaluated_forecast_days,
        minimum_required=decision.minimum_required,
        cooldown_days=decision.cooldown_days,
        cooldown_remaining_days=decision.cooldown_remaining_days,
        baseline_model=baseline_model,
        source_monitoring_snapshot_id=decision.source_monitoring_snapshot_id,
        last_retraining_attempt=(
            decision.last_retraining_attempt_at.isoformat()
            if decision.last_retraining_attempt_at is not None
            else None
        ),
        retraining_run_id=decision.retraining_run.id if decision.retraining_run is not None else None,
        automatic_execution_enabled=decision.automatic_execution_enabled,
    )


@app.get(
    "/api/model-monitoring",
    dependencies=[Depends(verify_api_key)],
    response_model=ModelMonitoringSnapshotResponse,
)
async def current_model_monitoring(
    monitoring_service: ModelMonitoringService = Depends(get_model_monitoring_service),
):
    """Return the latest model monitoring snapshot for the current model scope."""
    try:
        snapshot = monitoring_service.current_snapshot()
    except SQLAlchemyError as exc:
        logger.exception("Model monitoring lookup failed")
        raise HTTPException(status_code=503, detail="Model monitoring persistence is unavailable.") from exc
    if snapshot is None:
        return _unavailable_monitoring_response()
    return _monitoring_response(snapshot)


@app.get(
    "/api/model-monitoring/history",
    dependencies=[Depends(verify_api_key)],
    response_model=ModelMonitoringHistoryResponse,
)
async def model_monitoring_history(
    limit: int = Query(20, ge=1, le=100),
    model_artifact_id: Optional[int] = Query(None, ge=1),
    status: Optional[PersistedMonitoringStatus] = None,
    monitoring_service: ModelMonitoringService = Depends(get_model_monitoring_service),
):
    """Return recent monitoring snapshots, newest first."""
    try:
        rows = monitoring_service.snapshot_history(
            limit=limit,
            model_artifact_id=model_artifact_id,
            status=status,
        )
    except SQLAlchemyError as exc:
        logger.exception("Model monitoring history lookup failed")
        raise HTTPException(status_code=503, detail="Model monitoring persistence is unavailable.") from exc
    items = [_monitoring_response(row) for row in rows]
    return ModelMonitoringHistoryResponse(items=items, limit=limit, count=len(items))


@app.post(
    "/api/model-monitoring/evaluate",
    dependencies=[Depends(verify_api_key)],
    response_model=ModelMonitoringSnapshotResponse,
)
async def evaluate_model_monitoring(
    monitoring_service: ModelMonitoringService = Depends(get_model_monitoring_service),
):
    """Create or reuse one monitoring snapshot; does not retrain or promote."""
    try:
        result = monitoring_service.create_snapshot()
    except SQLAlchemyError as exc:
        logger.exception("Model monitoring evaluation failed")
        raise HTTPException(status_code=503, detail="Model monitoring persistence is unavailable.") from exc
    return _monitoring_response(result.snapshot, created=result.created)


@app.get(
    "/api/model-retraining/status",
    dependencies=[Depends(verify_api_key)],
    response_model=ModelRetrainingStatusResponse,
)
async def model_retraining_status(
    retraining_service: RetrainingDecisionService = Depends(get_retraining_decision_service),
):
    """Return read-only retraining recommendation status.

    This endpoint does not start training, register candidates, promote models,
    or mutate the recommendation table.
    """
    try:
        decision = retraining_service.evaluate(persist_recommendation=False)
    except SQLAlchemyError as exc:
        logger.exception("Retraining recommendation lookup failed")
        raise HTTPException(status_code=503, detail="Retraining recommendation persistence is unavailable.") from exc
    return _retraining_status_response(decision)


_KPI_INTERPRETATION = {
    "baseline": "naive",
    "baseline_description": (
        "Fixed-threshold policy: reorder 2 weeks of average demand whenever "
        "stock drops below 1 week of average demand."
    ),
    "intelligent_description": (
        "Adaptive per-SKU policy: demand classified (regular / intermittent / "
        "highly-intermittent), forecast produced by LightGBM, Croston, or a "
        "conservative buffer, dynamic safety stock from rolling forecast error, "
        "reorder point = lead-time demand + safety stock."
    ),
    "assumptions": {
        "lead_time_days": 7,
        "service_level": 0.95,
        "holding_cost_per_unit": 0.5,
        "stockout_cost_per_unit": 5.0,
        "simulation_window_days": 90,
    },
    "metric_meanings": {
        "cost_savings_pct": "Total cost (holding + stockout) saved by the intelligent policy relative to the naive baseline, aggregated across simulated SKUs.",
        "fill_rate": "Fraction of demanded units that were actually fulfilled under the intelligent policy; higher is better, 1.0 = no stockouts.",
        "naive_total_cost": "Simulated total cost (holding + stockout) of the naive baseline policy.",
        "intelligent_total_cost": "Simulated total cost of the intelligent policy on the same SKUs and time window.",
        "holding_cost": "Cost of units held in inventory across the simulation window.",
        "stockout_cost": "Penalty incurred for unmet demand across the simulation window.",
        "skus_analyzed": "How many SKUs actually produced usable simulations (>= 60 days of history and non-zero demand).",
    },
}


@app.get("/api/kpis", dependencies=[Depends(verify_api_key)])
async def kpis():
    cache_path = Path(__file__).resolve().parent / "data" / "cached_kpis.json"
    if not cache_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "KPIs not computed. Run: cd backend && python scripts/compute_kpis.py "
                "(or `python scripts/bootstrap.py` to generate everything at once)."
            ),
        )
    with open(cache_path) as f:
        payload = json.load(f)
    # Attach interpretation metadata so a consumer can tell what each number
    # means without cross-referencing the README.
    payload["interpretation"] = _KPI_INTERPRETATION
    return payload


@app.get("/api/skus", dependencies=[Depends(verify_api_key)])
async def list_skus():
    if _data_service is None:
        raise HTTPException(status_code=503, detail=_MISSING_DATA_MSG)
    return {"skus": _data_service.get_top_skus(n=20)}


@app.get("/api/skus/details", dependencies=[Depends(verify_api_key)])
async def skus_with_details():
    if _data_service is None:
        raise HTTPException(status_code=503, detail=_MISSING_DATA_MSG)
    top_skus = _data_service.get_top_skus(n=20)
    result = []
    for sku_code in top_skus:
        demand = _data_service.get_demand_history(sku_code)
        avg_demand = float(demand.mean()) if len(demand) > 0 else 0.0
        total_demand = float(demand.sum()) if len(demand) > 0 else 0.0
        result.append({
            "id": sku_code,
            "name": _sku_descriptions.get(sku_code, f"SKU {sku_code}"),
            "avg_demand": round(avg_demand, 1),
            "total_demand": round(total_demand),
        })
    return {"skus": result}


def _stock_error(detail: str, status_code: int = 503) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


@app.get(
    "/api/stock",
    dependencies=[Depends(verify_api_key)],
    response_model=StockListResponse,
)
async def list_stock(stock_service: StockService = Depends(get_stock_service)):
    """Return latest server-side stock snapshots for all SKUs with stock."""
    try:
        snapshots = stock_service.list_latest_stock()
    except StockPersistenceUnavailableError as exc:
        raise _stock_error(str(exc)) from exc
    return {"items": [snapshot.__dict__ for snapshot in snapshots], "source": "database"}


@app.get(
    "/api/stock/{sku_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=StockResponse,
)
async def get_stock(sku_id: str, stock_service: StockService = Depends(get_stock_service)):
    """Return the latest server-side stock snapshot for one SKU."""
    try:
        snapshot = stock_service.get_latest_stock(sku_id)
    except StockPersistenceUnavailableError as exc:
        raise _stock_error(str(exc)) from exc
    if snapshot is None:
        raise _stock_error(f"No server-side stock recorded for SKU {sku_id}.", status_code=404)
    return snapshot.__dict__


@app.put(
    "/api/stock/{sku_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=StockResponse,
)
async def update_stock(
    sku_id: str,
    body: StockUpdateRequest,
    stock_service: StockService = Depends(get_stock_service),
):
    """Append a server-side stock snapshot for one SKU."""
    try:
        snapshot = stock_service.record_stock(
            sku_code=sku_id,
            quantity_on_hand=body.quantity_on_hand,
            quantity_reserved=body.quantity_reserved,
            note=body.note,
            sku_name=body.sku_name,
        )
    except InvalidStockLevelError as exc:
        raise _stock_error(str(exc), status_code=422) from exc
    except StockPersistenceUnavailableError as exc:
        raise _stock_error(str(exc)) from exc
    return snapshot.__dict__


_HISTORY_METADATA = {
    "series_type": "recorded_history",
    "value_meaning": "actual_units_sold",
    "source": "processed_dataset",
    "description": (
        "Each entry is the real number of units sold on that calendar day, "
        "taken directly from the processed sales dataset. These are not "
        "forecasts — forecast values live on POST /api/analyze."
    ),
}


@app.get("/api/skus/{sku}/history", dependencies=[Depends(verify_api_key)])
async def sku_history(sku: str, days: int = 30):
    """Return the last ``days`` of recorded actual sales for a SKU.

    Each row is the real units-sold count for one calendar day from the
    processed daily-demand dataset — no synthesis or imputation beyond the
    date-reindex that already runs inside ``DataService.get_demand_history``.
    The response envelope carries provenance metadata (``series_type``,
    ``value_meaning``, ``source``) plus a lightweight ``summary`` so the UI
    can label the values unambiguously as past actuals, not forecasts.

    When the SKU has no recorded history the response carries an empty
    ``history`` list and ``available=False`` so the UI can render an honest
    empty state.
    """
    if _data_service is None:
        raise HTTPException(status_code=503, detail=_MISSING_DATA_MSG)

    days = max(1, min(int(days), 365))

    series = _data_service.get_demand_history(sku)
    if len(series) == 0:
        return {
            "sku": sku,
            "available": False,
            "history": [],
            "summary": None,
            "window_days_requested": days,
            **_HISTORY_METADATA,
        }

    tail = series.tail(days)
    # Each row retains the legacy ``demand`` key for backward compat, and
    # exposes ``units_sold`` as the preferred, unambiguous name going forward.
    history = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "demand": float(val),
            "units_sold": float(val),
        }
        for idx, val in tail.items()
    ]
    values = [row["units_sold"] for row in history]
    days_with_sales = sum(1 for v in values if v > 0)
    summary = {
        "first_date": history[0]["date"],
        "last_date": history[-1]["date"],
        "window_days_returned": len(history),
        "total_units_sold": round(sum(values), 2),
        "mean_units_per_day": round(sum(values) / len(values), 2) if values else 0.0,
        "peak_units_in_one_day": round(max(values), 2) if values else 0.0,
        "days_with_sales": days_with_sales,
        "days_with_zero_sales": len(values) - days_with_sales,
    }
    return {
        "sku": sku,
        "available": True,
        "history": history,
        "summary": summary,
        "window_days_requested": days,
        **_HISTORY_METADATA,
    }


@app.post(
    "/api/analyze",
    dependencies=[Depends(verify_api_key)],
    response_model=AnalyzeResponse,
)
async def analyze_sku(
    body: AnalyzeRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
):
    """Risk assessment + reorder recommendation for a single SKU."""
    try:
        return analysis_service.analyze(body)
    except AnalysisServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Inventory service unavailable")
    except AnalysisExecutionError as exc:
        logger.exception("Analysis failed for %s", body.sku)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
