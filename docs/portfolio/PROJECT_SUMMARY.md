# Project Summary — SupplySync AI

Reusable descriptions for GitHub, resumes, LinkedIn, and job applications.

## One-Sentence Version

SupplySync AI is an ML-powered inventory decision-support prototype that routes SKUs to a hybrid forecasting method by demand pattern and manages its own model lifecycle through a controlled, human-approved MLOps pipeline.

## Three-Sentence Recruiter Version

SupplySync AI turns historical retail demand into inventory reorder recommendations by classifying each SKU's demand pattern and routing it to the forecasting method suited to that pattern — LightGBM for regular demand, Croston-SBA for intermittent demand, a conservative buffer for highly sparse demand. Forecasts are converted into reorder decisions using uncertainty-aware safety stock and real supplier constraints (MOQ, order multiples, order caps), with every recommendation and prediction persisted for later evaluation. The project also implements a controlled MLOps lifecycle — performance monitoring, degradation detection, retraining recommendations, candidate evaluation, and human-approved promotion/rollback — deployed full-stack (Next.js, FastAPI, PostgreSQL) with 280+ automated tests in CI.

## Technical Reviewer Version

SupplySync AI is a full-stack inventory decision-support system built on the UCI Online Retail II dataset (~4,900 SKUs, ~531K daily-demand records, 2009–2011). Demand is classified by zero-demand share into regular/intermittent/highly-intermittent buckets, each routed to a distinct forecasting method: a 15-feature LightGBM model (7 lags, 3 rolling statistics, 5 calendar features) for regular demand, Croston's method with Syntetos-Boylan bias correction for intermittent demand, and a conservative recent-mean buffer for highly intermittent demand — validated against four classical baselines in an offline temporal-holdout backtest rather than assumed. Forecasts feed a reorder-point calculation (lead-time demand + Z-score or residual-based dynamic safety stock) constrained by MOQ/order-multiple/max-order rules, with SKU-specific policy overrides taking precedence over defaults. Every analysis persists an `analysis_runs` row and a linked `prediction_logs` row through SQLAlchemy repositories; once a prediction's target window completes, it is scored (WAPE/MAE/RMSE/Bias/MASE) with explicit leakage guards, rolled into monitoring snapshots that classify recent performance as stable/warning/degraded against a baseline, and can trigger a retraining recommendation. Candidate models are trained and evaluated against an evidence gate (checksum, feature-schema match, minimum test points, horizon compatibility, bias safety) before a human operator can promote or roll back the serving artifact through a checksum/schema/deserialization-validated, fully audited CLI — automatic retraining and automatic promotion are both explicitly disabled by configuration. Because the dataset is historical and frozen, live monitoring cannot accumulate genuinely new evidence; a separate Historical Monitoring Replay mechanism demonstrates the identical evaluate-monitor-classify pipeline against held-out historical windows, clearly labeled as replay and structurally isolated from the tables live retraining decisions read. The system is deployed as Next.js (Vercel) + FastAPI (Render) + PostgreSQL (Neon), with Alembic-managed migrations validated in CI against a real PostgreSQL service container alongside 280 backend and 30 frontend tests.

## Key Achievements

- Evidence-driven hybrid forecasting router validated by an offline backtest, not assumption (Croston-SBA measurably beats LightGBM in aggregate on this dataset — and the system routes accordingly instead of hiding it).
- A genuinely controlled MLOps lifecycle: monitoring, degradation detection, retraining recommendation, candidate evaluation, and promotion/rollback, all human-gated and fully audited.
- A historical-replay mechanism built specifically to avoid fabricating live monitoring evidence on a frozen dataset — an honest solution to a real constraint rather than a workaround that hides it.
- A real, found-and-fixed concurrency bug in the model-promotion rollback path (a race that could momentarily violate the one-active-artifact database invariant).
- 280+ backend tests and 30 frontend tests, with CI validating a full PostgreSQL migration round-trip, not just SQLite.

## Technology List

Python, FastAPI, SQLAlchemy, Pydantic, Alembic, PostgreSQL, LightGBM, pandas, NumPy, scikit-learn, SciPy, PyArrow/Parquet, Next.js, React, TypeScript, GitHub Actions, Render, Vercel, Neon.

## Limitations (state plainly, do not omit)

- Demo dataset is historical and frozen; no live ERP/POS integration exists.
- Live production monitoring has no new evidence to accumulate without a live demand feed — historical replay is a deliberate substitute, not a replacement, and is never presented as live evidence.
- Recursive multi-step LightGBM forecasting does not currently advance future calendar features correctly — a known, deferred limitation.
- Forecast-performance monitoring exists; feature/input-distribution drift detection does not.
- Automatic retraining and automatic promotion are both intentionally disabled — every model lifecycle change requires an explicit human command.
- Production assumes a single backend worker; a promotion requires a restart/redeploy to take effect.

This is a decision-support prototype, not autonomous purchasing, a live ERP, a real-time POS system, or a self-healing AI system.
