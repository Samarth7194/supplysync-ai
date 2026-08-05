# backend/main.py

import hashlib
import hmac
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure 'src' is accessible for intra-package imports like `from services...`
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from services.data_service import DataService
from services.intelligent_inventory_service import IntelligentInventoryService
from services.model_service import get_model_service
from ingestion.load_retail_data import get_sku_descriptions
from storage.analysis_store import AnalysisStore, serialize_rows
from config.settings import load_settings
from dependencies.stock import get_stock_service
from services.stock_service import (
    InvalidStockLevelError,
    StockPersistenceUnavailableError,
    StockService,
)
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
_analysis_store: Optional[AnalysisStore] = None

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
    global _loaded_model, _data_service, _sku_descriptions, _inventory_service, _analysis_store

    model_service = get_model_service(model_dir=SETTINGS.forecasting.model_path)
    model_feature_columns: Optional[List[str]] = None
    try:
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

    # Local/demo persistence. Gitignored; override path with ANALYSES_DB_PATH.
    try:
        _analysis_store = AnalysisStore(SETTINGS.database.analyses_db_path)
        logger.info("Analysis store ready at %s", SETTINGS.database.analyses_db_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("Analysis store unavailable: %s", e)
        _analysis_store = None

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


class ModelInfo(BaseModel):
    """Compact model/method provenance.

    Always truthful: for statistical and rule-based paths ``model_type`` and
    ``model_name`` reflect the actual method used, not a hallucinated ML model.
    """
    model_name: str
    model_type: str                  # "ml" | "statistical_method" | "rule_based_fallback" | "none"
    artifact_available: bool         # whether a saved model file is loaded
    trained_at: Optional[str] = None
    feature_count: Optional[int] = None
    dataset: Optional[str] = None
    evaluation_available: bool = False
    evaluation_generated_at: Optional[str] = None


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


# --- Helpers ---

def _stable_hash_int(s: str) -> int:
    """Process-stable hash; Python's built-in hash() is salted per-process."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "big")


# Classifies a concrete forecast method into a provenance bucket the UI can
# badge consistently. Kept as an explicit map so new methods must be added
# deliberately.
_FORECAST_SOURCE_BY_METHOD = {
    "ml_lightgbm": "model_forecast",
    "croston": "statistical_method",
    "conservative": "statistical_method",
    "simple_average": "rule_based_estimate",
}


def _classify_forecast_source(method: str) -> str:
    return _FORECAST_SOURCE_BY_METHOD.get(method, "unavailable")


# --- Explanation helpers ----------------------------------------------------


def _compose_explanation(
    *,
    demand_series: pd.Series,
    demand_pattern: str,
    forecast_method: str,
    forecast_source: str,
    demand_source: str,
    risk: str,
    p50: float,
    p90: float,
    current_stock: float,
    model_artifact_loaded: bool,
) -> ExplanationBlock:
    """Produce short, truthful explanations tied to real inputs.

    All text is composed from observable inputs — demand series stats,
    routing decisions, model-loaded state, stock vs P50/P90. No LLM, no
    fabricated confidence number.
    """
    arr = demand_series.to_numpy(dtype=float)
    zero_share = float((arr == 0).mean()) if arr.size else 0.0
    n_obs = int(arr.size)

    # 1) Why this demand pattern?
    if demand_pattern == "highly_intermittent":
        classification_reason = (
            f"{zero_share * 100:.0f}% of the {n_obs} observed days have zero demand, "
            "which crosses the 80% threshold for the highly-intermittent class."
        )
    elif demand_pattern == "intermittent":
        classification_reason = (
            f"{zero_share * 100:.0f}% of the {n_obs} observed days have zero demand, "
            "which falls between the 50% and 80% thresholds used for the intermittent class."
        )
    else:
        classification_reason = (
            f"Only {zero_share * 100:.0f}% of the {n_obs} observed days have zero demand "
            "(below the 50% threshold), so this SKU is routed as regular demand."
        )

    # 2) Why this method?
    if forecast_method == "ml_lightgbm":
        method_reason = (
            "Regular-demand SKUs are forecast with the trained LightGBM model using "
            "lag and calendar features — that's the path chosen here."
        )
    elif forecast_method == "croston":
        method_reason = (
            "Intermittent SKUs are forecast with Croston's method (SBA-corrected), "
            "which separately estimates demand size and inter-arrival interval."
        )
    elif forecast_method == "conservative":
        method_reason = (
            "Highly-intermittent SKUs use a conservative buffer (recent mean × 1.5) "
            "because few non-zero days make any model's point forecast unreliable."
        )
    elif forecast_method == "simple_average":
        if not model_artifact_loaded:
            method_reason = (
                "The regular-demand SKU would normally be forecast by LightGBM, but "
                "the trained artifact isn't loaded on this deployment — falling back "
                "to a 7-day moving average. Treat the recommendation as approximate."
            )
        else:
            method_reason = (
                "The regular-demand SKU fell back to a 7-day moving average "
                "(likely due to insufficient history or a feature-build failure); "
                "see server logs for the exact reason."
            )
    else:
        method_reason = f"Forecast method {forecast_method!r} was used."

    # 3) Why this risk bucket?
    if risk == "HIGH":
        risk_reason = (
            f"Current stock ({current_stock:g}) is below the P50 demand estimate "
            f"({p50:g}) — any higher-than-median day risks a stockout."
        )
    elif risk == "MEDIUM":
        risk_reason = (
            f"Current stock ({current_stock:g}) covers median demand ({p50:g}) but not "
            f"the P90 scenario ({p90:g}) — a higher-demand day could cause a stockout."
        )
    else:
        risk_reason = (
            f"Current stock ({current_stock:g}) covers the P90 demand scenario "
            f"({p90:g}), so even a higher-demand day should be fulfillable."
        )

    # 4) Confidence caveat
    if demand_source == "synthetic":
        confidence_note = (
            "This SKU isn't in the processed dataset — the demand series is "
            "synthetic demo data, so the recommendation is illustrative only."
        )
    elif forecast_source == "rule_based_estimate":
        confidence_note = (
            "Forecast came from a rule-based fallback, not the trained model — "
            "the recommendation is coarser than the regular-path ML output."
        )
    elif forecast_source == "statistical_method":
        confidence_note = (
            "Forecast came from a statistical method tuned to sparse demand; "
            "point accuracy is inherently limited when most days have zero demand."
        )
    else:
        confidence_note = (
            "Forecast came from the trained LightGBM model; the recommendation "
            "reflects the model's regular-demand path."
        )

    return ExplanationBlock(
        classification_reason=classification_reason,
        method_reason=method_reason,
        risk_reason=risk_reason,
        confidence_note=confidence_note,
    )


# Human-friendly names for the non-ML paths. ML path keeps its artifact name.
_STATISTICAL_METHOD_NAMES = {
    "croston": "Croston (SBA-corrected)",
    "conservative": "Conservative buffer (recent mean × 1.5)",
}


def _model_info_for_method(method: str) -> ModelInfo:
    """Compose a ModelInfo block that truthfully describes the method used.

    For the LightGBM path we read artifact metadata from disk so the fields
    (``trained_at``, ``feature_count``, ``dataset``) reflect the live artifact.
    For statistical / rule-based paths we return the method name with no
    artifact attached.
    """
    backend_dir = Path(__file__).resolve().parent
    eval_path = backend_dir / "data" / "forecast_evaluation.json"
    eval_available = eval_path.exists()
    eval_generated_at: Optional[str] = None
    if eval_available:
        try:
            with eval_path.open() as fh:
                eval_generated_at = json.load(fh).get("generated_at")
        except Exception:  # noqa: BLE001 — evaluation file is non-critical
            eval_generated_at = None

    if method == "ml_lightgbm":
        model_service = get_model_service(
            model_dir=str(backend_dir / "saved_models"),
        )
        meta = model_service.get_model_metadata("lightgbm_demand_forecast") or {}
        features = meta.get("features") or []
        return ModelInfo(
            model_name="lightgbm_demand_forecast",
            model_type="ml",
            artifact_available=_loaded_model is not None,
            trained_at=meta.get("saved_at"),
            feature_count=len(features) if features else None,
            dataset=meta.get("dataset"),
            evaluation_available=eval_available,
            evaluation_generated_at=eval_generated_at,
        )

    if method in _STATISTICAL_METHOD_NAMES:
        return ModelInfo(
            model_name=_STATISTICAL_METHOD_NAMES[method],
            model_type="statistical_method",
            artifact_available=False,
            evaluation_available=eval_available,
            evaluation_generated_at=eval_generated_at,
        )

    if method == "simple_average":
        return ModelInfo(
            model_name="Simple 7-day moving average",
            model_type="rule_based_fallback",
            artifact_available=False,
            evaluation_available=eval_available,
            evaluation_generated_at=eval_generated_at,
        )

    return ModelInfo(
        model_name="unknown",
        model_type="none",
        artifact_available=False,
        evaluation_available=eval_available,
        evaluation_generated_at=eval_generated_at,
    )


def _compose_decision_why(
    *,
    order_qty: int,
    current_stock: float,
    reorder_point: float,
    lead_time_demand: float,
    lead_time_days: int,
    safety_stock: float,
    service_level: float,
) -> str:
    """Short human-readable sentence explaining the recommendation.

    Kept in main.py (not the service) because it's response-shape glue, not
    inventory logic. The numbers themselves are all already produced by
    ``compute_reorder_decision``.
    """
    sl_pct = int(round(service_level * 100))
    ltd = round(float(lead_time_demand), 1)
    ss = round(float(safety_stock), 1)
    rop = round(float(reorder_point), 1)
    stock = round(float(current_stock), 1)

    if order_qty > 0:
        return (
            f"Projected demand over the {lead_time_days}-day lead time is {ltd} units. "
            f"With a {ss}-unit safety buffer (targeting {sl_pct}% service level) the "
            f"reorder point is {rop}. Current stock {stock} is below that, so "
            f"ordering {order_qty} units brings the position back above the reorder point."
        )
    return (
        f"Projected demand over the {lead_time_days}-day lead time is {ltd} units. "
        f"Reorder point is {rop} (incl. {ss}-unit safety stock at {sl_pct}% service "
        f"level). Current stock {stock} already covers the reorder point — no action needed."
    )


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
    return {
        "status": "online",
        "model_loaded": _loaded_model is not None,
        "data_available": _data_service is not None,
        "kpis_available": kpi_cache.exists(),
        "hint": None if (_loaded_model and _data_service and kpi_cache.exists())
                else "Run `python scripts/bootstrap.py` to generate missing artifacts.",
    }


@app.get("/api/analyses/recent", dependencies=[Depends(verify_api_key)])
async def recent_analyses(limit: int = 20):
    """Return the most-recent persisted /api/analyze snapshots.

    Local/demo persistence only — the store is a SQLite file under
    ``backend/data/analyses.sqlite`` (override with ``ANALYSES_DB_PATH``).
    """
    if _analysis_store is None:
        return {"available": False, "items": [], "total": 0}
    rows = _analysis_store.recent(limit=limit)
    return {
        "available": True,
        "items": serialize_rows(rows),
        "total": _analysis_store.count(),
        "source": "sqlite (local/demo persistence)",
    }


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
async def analyze_sku(body: AnalyzeRequest):
    """Risk assessment + reorder recommendation for a single SKU."""
    if _inventory_service is None:
        raise HTTPException(status_code=503, detail="Inventory service unavailable")

    demand_source = "request"
    demand_series: Optional[pd.Series] = None

    if body.demand_history:
        demand_series = pd.Series(body.demand_history, dtype=float)
    elif _data_service is not None:
        hist = _data_service.get_demand_history(body.sku)
        if len(hist) > 0:
            # Keep the DatetimeIndex so inference features use real dates.
            demand_series = hist.tail(60).astype(float)
            demand_source = "historical"

    if demand_series is None or len(demand_series) == 0:
        # Deterministic synthetic series for demo/unknown SKUs. Flagged
        # in the response so callers can tell this wasn't real data.
        rng = np.random.default_rng(_stable_hash_int(body.sku))
        demand_series = pd.Series(rng.poisson(20, 30).astype(float))
        demand_source = "synthetic"

    effective_lead_time = (
        int(body.lead_time_days)
        if body.lead_time_days
        else SETTINGS.inventory.default_lead_time_days
    )
    effective_service_level = (
        float(body.service_level)
        if body.service_level
        else SETTINGS.inventory.default_service_level
    )
    try:
        decision = _inventory_service.get_intelligent_reorder_decision(
            sku=body.sku,
            current_stock=body.current_stock,
            demand_history=demand_series,
            lead_time_days=effective_lead_time,
            service_level=effective_service_level,
        )
    except Exception as e:
        logger.exception("Analysis failed for %s", body.sku)
        raise HTTPException(status_code=500, detail=f"Analysis failed for SKU {body.sku}: {e}")

    demand_values = demand_series.to_numpy(dtype=float)
    p50 = float(np.mean(demand_values))
    p90 = float(np.percentile(demand_values, 90))

    if body.current_stock < p50:
        risk, risk_color = "HIGH", "#ef4444"
    elif body.current_stock < p90:
        risk, risk_color = "MEDIUM", "#eab308"
    else:
        risk, risk_color = "LOW", "#22c55e"

    order_qty = int(decision.get("order_quantity", 0))
    forecast_daily_raw = decision.get("forecast_daily") or []
    if forecast_daily_raw:
        daily_forecast = [round(float(v), 2) for v in forecast_daily_raw[:7]]
        if len(daily_forecast) < 7:
            daily_forecast += [daily_forecast[-1]] * (7 - len(daily_forecast))
    else:
        lead_time_demand = float(decision.get("lead_time_demand", 0))
        daily_forecast = [round(lead_time_demand / 7, 2)] * 7

    forecast_method = decision.get("intelligence", {}).get("forecast_method", "unknown")

    lead_time_demand = float(decision.get("lead_time_demand", 0.0))
    safety_stock = float(decision.get("safety_stock", 0.0))
    safety_stock_method = str(decision.get("safety_stock_method", "traditional"))
    reorder_point = float(decision.get("reorder_point", 0.0))
    service_level = float(decision.get("service_level", 0.95))
    lead_time_days_v = int(decision.get("lead_time_days", 7))
    inventory_gap = max(0.0, reorder_point - float(body.current_stock))

    decision_block = DecisionBlock(
        lead_time_days=lead_time_days_v,
        lead_time_demand=round(lead_time_demand, 2),
        safety_stock=round(safety_stock, 2),
        safety_stock_method=safety_stock_method,
        reorder_point=round(reorder_point, 2),
        service_level=service_level,
        inventory_gap=round(inventory_gap, 2),
        why=_compose_decision_why(
            order_qty=order_qty,
            current_stock=body.current_stock,
            reorder_point=reorder_point,
            lead_time_demand=lead_time_demand,
            lead_time_days=lead_time_days_v,
            safety_stock=safety_stock,
            service_level=service_level,
        ),
    )

    demand_pattern_v = decision.get("intelligence", {}).get("demand_pattern", "unknown")
    forecast_source_v = _classify_forecast_source(forecast_method)
    model_info_v = _model_info_for_method(forecast_method)
    explanation_v = _compose_explanation(
        demand_series=demand_series,
        demand_pattern=demand_pattern_v,
        forecast_method=forecast_method,
        forecast_source=forecast_source_v,
        demand_source=demand_source,
        risk=risk,
        p50=p50,
        p90=p90,
        current_stock=float(body.current_stock),
        model_artifact_loaded=_loaded_model is not None,
    )

    response = AnalyzeResponse(
        sku=body.sku,
        risk=risk,
        risk_color=risk_color,
        forecast=ForecastBlock(p50=round(p50, 1), p90=round(p90, 1), daily=daily_forecast),
        current_stock=body.current_stock,
        recommended_order=order_qty,
        action="PURCHASE" if order_qty > 0 else "NO_ACTION",
        demand_pattern=demand_pattern_v,
        forecast_method=forecast_method,
        demand_source=demand_source,
        forecast_source=forecast_source_v,
        decision=decision_block,
        model_info=model_info_v,
        explanation=explanation_v,
    )

    # Best-effort persistence (never fails the live request).
    if _analysis_store is not None:
        try:
            _analysis_store.record({
                "sku": response.sku,
                "risk": response.risk,
                "action": response.action,
                "current_stock": response.current_stock,
                "recommended_order": response.recommended_order,
                "demand_pattern": response.demand_pattern,
                "forecast_method": response.forecast_method,
                "demand_source": response.demand_source,
                "forecast_source": response.forecast_source,
                "model_type": model_info_v.model_type,
                "model_name": model_info_v.model_name,
                "artifact_available": model_info_v.artifact_available,
                "lead_time_demand": decision_block.lead_time_demand,
                "safety_stock": decision_block.safety_stock,
                "reorder_point": decision_block.reorder_point,
                "inventory_gap": decision_block.inventory_gap,
                "p50": response.forecast.p50,
                "p90": response.forecast.p90,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Analysis persistence failed for %s: %s", body.sku, exc)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
