# API Reference

Short, real examples for the endpoints you'll actually hit. See the root [README](../README.md) for setup and [docs/architecture.md](./architecture.md) for how the pieces fit.

---

## Base URL & auth

| | |
|---|---|
| Base URL | `http://localhost:8000` (configurable via `NEXT_PUBLIC_API_URL` on the frontend) |
| Auth modes | `AUTH_MODE=off` (default) — open API. `AUTH_MODE=demo` — requires login via `POST /api/auth/login` (session cookie) **or** the legacy `X-API-Key` header. |
| Content type | `application/json` for requests that have a body |

Both modes keep `/health` and `/api/auth/*` public. Browser clients should send credentials with every fetch (`credentials: "include"`) so the session cookie crosses origins during dev.

All examples below omit auth; add the cookie or `X-API-Key` header when `AUTH_MODE=demo`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`                      | Liveness + readiness flags (`model_loaded`, `data_available`, `kpis_available`, `hint`) |
| GET  | `/api/model-info`              | Trained-artifact metadata + evaluation pointer |
| GET  | `/api/kpis`                    | Simulated cost/fill-rate KPIs with an `interpretation` block |
| GET  | `/api/skus`                    | Top-20 SKU codes |
| GET  | `/api/skus/details`            | Top-20 SKUs with `avg_demand` / `total_demand` |
| GET  | `/api/stock`                   | Latest server-side stock snapshots for SKUs with recorded stock |
| GET  | `/api/stock/{sku_id}`          | Latest server-side stock snapshot for one SKU |
| PUT  | `/api/stock/{sku_id}`          | Append a server-side stock snapshot for one SKU |
| GET  | `/api/skus/{sku}/history?days=30` | Recorded daily demand for a SKU |
| POST | `/api/analyze`                 | Risk + reorder recommendation for a single SKU |
| GET  | `/api/analyses/recent?limit=N`    | Most-recent persisted analyses from SQLAlchemy `analysis_runs` |
| GET  | `/api/model-monitoring`        | Latest live monitoring snapshot for the active model |
| GET  | `/api/model-monitoring/history` | Recent monitoring snapshots, newest first |
| POST | `/api/model-monitoring/evaluate` | Create/reuse one monitoring snapshot — never retrains or promotes |
| GET  | `/api/model-monitoring/replay` | Read-only historical monitoring replay evidence — never live; see [docs/mlops-monitoring.md](./mlops-monitoring.md) |
| GET  | `/api/model-retraining/status` | Read-only retraining recommendation status — never persists or trains |

The MLOps endpoints above are documented in depth in [docs/mlops-monitoring.md](./mlops-monitoring.md) and [docs/model-promotion.md](./model-promotion.md) rather than repeated here — promotion and rollback are deliberately CLI-only and have no HTTP route.

---

## `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "online",
  "model_loaded": true,
  "data_available": true,
  "kpis_available": true,
  "hint": null
}
```

If any of the readiness flags is `false`, `hint` becomes a human-readable string that names the fix (e.g. `"Run python scripts/bootstrap.py to generate missing artifacts."`). Safe to poll as a Docker healthcheck.

---

## `GET /api/model-info`

```bash
curl http://localhost:8000/api/model-info
```

```json
{
  "model_name": "lightgbm_demand_forecast",
  "model_type": "ml",
  "artifact_available": true,
  "model_version": "lightgbm_demand_forecast-20260523T084002Z-ea7815fa9a33",
  "feature_schema_version": "demand_lag_calendar_v1",
  "artifact_valid": true,
  "lifecycle_status": "candidate",
  "trained_at": "2026-04-21T13:35:21+00:00",
  "dataset": "online_retail_II.csv",
  "feature_count": 15,
  "features": [
    "lag_1","lag_2","lag_3","lag_4","lag_5","lag_6","lag_7",
    "rolling_mean_7","rolling_std_7","rolling_mean_14",
    "day_of_week","month","is_weekend","day_of_month","week_of_year"
  ],
  "train_skus": ["84077","85099B","21212", "..."],
  "training_metrics": { "mae": 91.67, "rmse": 191.12 },
  "evaluation": {
    "available": true,
    "generated_at": "2026-04-21T13:37:45.XXXXXX",
    "summary": { "mae": 91.67, "rmse": 143.58, "bias": 26.06, "wape": 0.99, "mase": 0.83, "n_skus": 20, "n_test_points": 600 }
  },
  "hint": null
}
```

When the `.pkl` isn't present, `artifact_available` is `false` and `hint` is set to an actionable message:

```json
{
  "model_name": "lightgbm_demand_forecast",
  "model_type": "ml",
  "artifact_available": false,
  "trained_at": null,
  "dataset": null,
  "feature_count": null,
  "features": null,
  "train_skus": null,
  "training_metrics": { "mae": null, "rmse": null },
  "evaluation": { "available": false, "generated_at": null, "summary": null },
  "hint": "Trained model not loaded. Run `python scripts/bootstrap.py` to build the artifact; until then regular-demand forecasts fall back to the 7-day moving average."
}
```

---

## `GET /api/kpis`

```bash
curl http://localhost:8000/api/kpis
```

```json
{
  "total_cost": 1647932.0,
  "fill_rate": 0.9567,
  "cost_savings_pct": 37.8,
  "holding_cost": 1624917.0,
  "stockout_cost": 23015.0,
  "naive_total_cost": 2650120.0,
  "intelligent_total_cost": 1647932.0,
  "skus_analyzed": 10,
  "computed_at": "2026-04-01T13:18:17.588264",
  "interpretation": {
    "baseline": "naive",
    "baseline_description": "Fixed-threshold policy: reorder 2 weeks of average demand whenever stock drops below 1 week of average demand.",
    "intelligent_description": "Adaptive per-SKU policy: ...",
    "assumptions": {
      "lead_time_days": 7,
      "service_level": 0.95,
      "holding_cost_per_unit": 0.5,
      "stockout_cost_per_unit": 5.0,
      "simulation_window_days": 90
    },
    "metric_meanings": {
      "cost_savings_pct": "Total cost (holding + stockout) saved by the intelligent policy relative to the naive baseline...",
      "fill_rate": "Fraction of demanded units that were actually fulfilled...",
      "..." : "..."
    }
  }
}
```

Returns **404** when the cache hasn't been generated:

```json
{
  "detail": "KPIs not computed. Run: cd backend && python scripts/compute_kpis.py (or `python scripts/bootstrap.py` to generate everything at once)."
}
```

---

## `GET /api/skus/{sku}/history?days=30`

```bash
curl "http://localhost:8000/api/skus/85099B/history?days=5"
```

```json
{
  "sku": "85099B",
  "available": true,
  "history": [
    { "date": "2011-12-05", "demand": 18 },
    { "date": "2011-12-06", "demand": 24 },
    { "date": "2011-12-07", "demand": 12 },
    { "date": "2011-12-08", "demand": 0 },
    { "date": "2011-12-09", "demand": 31 }
  ]
}
```

For unknown SKUs, `available` is `false` and `history` is an empty array — never fabricated:

```json
{ "sku": "NOT-A-REAL-SKU", "available": false, "history": [] }
```

`days` is clamped to `[1, 365]`.

---

## Server-side stock

`GET /api/stock` returns the latest server-side stock snapshot for every SKU
with recorded stock. `GET /api/stock/{sku_id}` returns one SKU, and
`PUT /api/stock/{sku_id}` appends a new stock snapshot.

Stock writes are append-only so historical recommendations remain auditable.
`/api/analyze` still accepts `current_stock` in the request body for backward
compatibility; the frontend now prefers the stock API and falls back to browser
storage only if the stock endpoint is unavailable or no server-side value exists.

Example write:

```json
{
  "quantity_on_hand": 80,
  "quantity_reserved": 5,
  "note": "manual count after receiving shipment"
}
```

Example response:

```json
{
  "sku": "85099B",
  "quantity_on_hand": 80.0,
  "quantity_reserved": 5.0,
  "quantity_available": 75.0,
  "source": "user",
  "recorded_at": "2026-08-05T08:15:00+00:00"
}
```

If the SQLAlchemy database has not been migrated yet, stock endpoints return
503 with an actionable migration message.

---

## `POST /api/analyze`

The main decision endpoint. Takes a SKU + current stock, returns the full analysis.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"sku":"85099B","current_stock":50}'
```

```json
{
  "sku": "85099B",
  "risk": "HIGH",
  "risk_color": "#ef4444",
  "forecast": {
    "p50": 85.2,
    "p90": 268.1,
    "daily": [79.1, 82.5, 88.0, 84.3, 91.2, 85.9, 82.7],
    "full_horizon_daily": [79.1, 82.5, 88.0, 84.3, 91.2, 85.9, 82.7],
    "horizon_days": 7
  },
  "current_stock": 50,
  "recommended_order": 740,
  "action": "PURCHASE",
  "demand_pattern": "regular",
  "forecast_method": "ml_lightgbm",
  "demand_source": "historical",
  "forecast_source": "model_forecast",
  "decision": {
    "lead_time_days": 7,
    "lead_time_demand": 593.7,
    "safety_stock": 194.3,
    "safety_stock_method": "traditional",
    "reorder_point": 788.0,
    "service_level": 0.95,
    "inventory_gap": 738.0,
    "constraints": {
      "raw_order_quantity": 738,
      "final_order_quantity": 740,
      "moq": 10,
      "order_multiple": 5,
      "max_order_quantity": 1000,
      "constraints_applied": ["Rounded to multiple of 5"],
      "moq_applied": false,
      "order_multiple_applied": true,
      "max_order_cap_applied": false,
      "constrained": true,
      "remaining_gap_after_order": 0.0
    },
    "why": "Projected demand over the 7-day lead time is 593.7 units. With a 194.3-unit safety buffer (targeting 95% service level) the reorder point is 788.0. Current stock 50.0 is below that, so ordering 740 units brings the position back above the reorder point."
  },
  "model_info": {
    "model_name": "lightgbm_demand_forecast",
    "model_type": "ml",
    "artifact_available": true,
    "trained_at": "2026-04-21T13:35:21+00:00",
    "feature_count": 15,
    "dataset": "online_retail_II.csv",
    "evaluation_available": true,
    "evaluation_generated_at": "2026-04-21T13:37:45.XXXXXX"
  },
  "explanation": {
    "classification_reason": "Only 3% of the 60 observed days have zero demand (below the 50% threshold), so this SKU is routed as regular demand.",
    "method_reason": "Regular-demand SKUs are forecast with the trained LightGBM model using lag and calendar features — that's the path chosen here.",
    "risk_reason": "Current stock (50) is below the P50 demand estimate (85.2) — any higher-than-median day risks a stockout.",
    "confidence_note": "Forecast came from the trained LightGBM model; the recommendation reflects the model's regular-demand path."
  }
}
```

(Numeric values vary by SKU, business-constraint configuration, and the trained model snapshot.)

### Interpreting the main blocks

| Block | What it tells you |
|---|---|
| `forecast` | Historical demand summary values used for risk classification, the legacy 7-value `daily` series, and `full_horizon_daily` for the requested lead-time horizon. |
| `decision` | Every number that goes into the reorder computation: lead-time demand, safety stock and its method, reorder point, inventory gap, business-constraint metadata, and a plain-English `why`. |
| `model_info` | Which path produced this response: trained ML, statistical method, or rule-based fallback. Honest about whether the `.pkl` is actually loaded. |
| `explanation` | Four short, deterministic strings (no LLM): *classification_reason*, *method_reason*, *risk_reason*, *confidence_note*. Useful for dashboards that want to show "why" without composing copy themselves. |
| `demand_source` | Where the input demand came from: `historical`, `request`, or `synthetic`. |
| `forecast_source` | Provenance of the forecast itself: `model_forecast`, `statistical_method`, `rule_based_estimate`, or `unavailable`. |

### Request options

| Field | Type | Notes |
|---|---|---|
| `sku` | string (required) | SKU identifier. Unknown SKUs trigger a deterministic synthetic fallback, flagged with `demand_source: "synthetic"`. |
| `current_stock` | number (≥ 0, default 50) | Used for the risk bucket and the reorder quantity. |
| `demand_history` | number[] *(optional)* | If provided, overrides the backend's dataset lookup and marks `demand_source: "request"`. |

### Degraded / fallback shapes

- **Unknown SKU**: `demand_source: "synthetic"`; all downstream numbers still produced but the frontend shows an amber banner and the chart is replaced with an empty state.
- **Model artifact missing**: `forecast_method: "simple_average"`, `forecast_source: "rule_based_estimate"`, `model_info.artifact_available: false`.
- **Intermittent SKU**: `forecast_method: "croston"` or `"conservative"`, `forecast_source: "statistical_method"`. The model is deliberately not used for sparse demand (see `docs/architecture.md` §4 and the evaluation in the README).

---

## `GET /api/analyses/recent?limit=10`

```bash
curl "http://localhost:8000/api/analyses/recent?limit=3"
```

```json
{
  "available": true,
  "source": "sqlalchemy",
  "total": 73,
  "items": [
    {
      "id": 73,
      "created_at": "2026-04-21T16:02:14+00:00",
      "sku": "85099B",
      "risk": "HIGH",
      "action": "PURCHASE",
      "current_stock": 50.0,
      "recommended_order": 740,
      "demand_pattern": "regular",
      "forecast_method": "ml_lightgbm",
      "demand_source": "historical",
      "forecast_source": "model_forecast",
      "model_type": "ml",
      "model_name": "lightgbm_demand_forecast",
      "artifact_available": true,
      "lead_time_demand": 593.7,
      "safety_stock": 194.3,
      "reorder_point": 788.0,
      "inventory_gap": 738.0,
      "p50": 85.2,
      "p90": 268.1
    }
  ]
}
```

Backed by SQLAlchemy `analysis_runs` and linked `prediction_logs`. `limit` is
clamped to `[1, 200]`. Local development can still use the default SQLite
`DATABASE_URL`, while Docker/CI/production should use PostgreSQL.

When the database is unavailable or migrations have not been applied:

```json
{ "detail": "Analysis history persistence is unavailable." }
```

---

## `/api/auth/*` (demo-mode login)

```bash
# Public — tells the UI whether a login gate is needed
curl http://localhost:8000/api/auth/status
```

```json
{
  "auth_mode": "demo",
  "authenticated": false,
  "user": null,
  "login_required": true,
  "note": "Demo auth: a single shared credential from DEMO_USER/DEMO_PASSWORD. Not a production identity system."
}
```

```bash
# Log in and capture the cookie
curl -c /tmp/cookies.txt -X POST http://localhost:8000/api/auth/login \
    -H 'content-type: application/json' \
    -d '{"username":"demo","password":"<DEMO_PASSWORD>"}'

# Use it
curl -b /tmp/cookies.txt http://localhost:8000/api/auth/me
# → { "user": "demo", "source": "session" }

curl -b /tmp/cookies.txt -X POST http://localhost:8000/api/auth/logout
# → { "ok": true }
```

The cookie is HMAC-SHA256 signed with `SESSION_SECRET`; default TTL is 12h; HttpOnly; SameSite=Lax. Deliberately a demo-grade guard — **not** a production identity system.

---

## Configurable planning assumptions

`POST /api/analyze` accepts two optional overrides:

| Field | Range | Default | Notes |
|---|---|---|---|
| `lead_time_days` | integer 1–90 | 7 | Days between placing an order and receiving it. |
| `service_level` | float (0.5, 1.0) exclusive | 0.95 | Target fraction of lead-time demand the safety buffer should cover. |

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H 'content-type: application/json' \
  -d '{"sku":"85099B","current_stock":50,"lead_time_days":14,"service_level":0.98}'
```

Higher `service_level` → larger `decision.safety_stock` → larger `decision.reorder_point` → potentially larger `recommended_order`. Longer `lead_time_days` → larger `decision.lead_time_demand` for the same forecast, same cascading effect. The SKU detail page uses these as sliders and re-runs the analysis live.

---

## Error responses

All error payloads follow FastAPI's default `{ "detail": "<string>" }` shape. The string is actionable — it names the fix or the missing dependency.

| Status | When | Example `detail` |
|---|---|---|
| 401 | `API_KEY` set but header missing or wrong | `"Invalid or missing API key"` |
| 404 | `/api/kpis` with no cache file | `"KPIs not computed. Run: cd backend && python scripts/compute_kpis.py ..."` |
| 500 | `/api/analyze` raised unexpectedly | `"Analysis failed for SKU <sku>: <exception message>"` |
| 503 | `/api/skus*` when the processed dataset isn't loaded | `"Data service unavailable — processed dataset missing. Generate it with: cd backend && python scripts/bootstrap.py"` |

---

## See also

- [docs/architecture.md](./architecture.md) — data flow + decision pipeline.
- [backend/scripts/evaluate_forecast.py](../backend/scripts/evaluate_forecast.py) — reproduce the `evaluation.summary` numbers the API surfaces.
- [backend/tests/](../backend/tests) — 75 pytest cases covering every endpoint's contract.

## Analysis Persistence Note

`POST /api/analyze` now persists through the application service layer. When
the SQLAlchemy database is migrated, each successful analysis writes:

```text
analysis_runs
  -> prediction_logs
```

`GET /api/analyses/recent` reads SQLAlchemy `analysis_runs`. There is no
secondary SQLite fallback in the FastAPI runtime.

Logged prediction evaluation is not part of request handling. Run
`python backend/scripts/evaluate_logged_predictions.py` to evaluate due
`prediction_logs` rows against recorded actual demand and persist
`forecast_evaluations` rows. This does not change the `/api/analyze` response
contract.

## Evidence Routing Note

`POST /api/analyze` remains backward compatible. The response fields are not
renamed or removed. Internally, `ModelRoutingService` may select among existing
forecasting methods when trustworthy evaluation evidence is available. The
selected executed method still appears in `forecast_method`, and the provenance
category still appears in `forecast_source`.

Routing provenance is persisted to `analysis_runs.routing_reason` and
`prediction_logs.routing_reason` when SQLAlchemy persistence is available.

