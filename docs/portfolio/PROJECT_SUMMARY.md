# Project Summary — SupplySync AI

Reusable descriptions for GitHub, resumes, LinkedIn, and job applications.

## One-Sentence Version

SupplySync AI is an ML-powered inventory decision-support prototype that routes SKUs to a hybrid forecasting method by demand pattern and manages its own model lifecycle through a controlled, human-approved MLOps pipeline.

## Three-Sentence Recruiter Version

SupplySync AI turns historical retail demand into inventory reorder recommendations by classifying each SKU's demand pattern and routing it, based on backtest evidence rather than assumption, to whichever forecasting method actually wins for that pattern — currently Croston-SBA for regular demand (LightGBM loses there), Croston-SBA for intermittent demand, and a conservative buffer for highly sparse demand. Forecasts are converted into reorder decisions using uncertainty-aware safety stock and real supplier constraints (MOQ, order multiples, order caps), with every recommendation and prediction persisted for later evaluation. The project also implements a controlled MLOps lifecycle — performance monitoring, degradation detection, retraining recommendations, candidate evaluation, and human-approved promotion/rollback — deployed full-stack (Next.js, FastAPI, PostgreSQL) with 360+ automated tests in CI.

## Technical Reviewer Version

SupplySync AI is a full-stack inventory decision-support system built on the UCI Online Retail II dataset (~4,900 SKUs, ~531K daily-demand records, 2009–2011; the live demo serves the top 20 by volume). Demand is classified by zero-demand share into regular/intermittent/highly-intermittent buckets, each routed to whichever forecasting method a shared, rolling-origin multi-step backtest actually shows winning for that class — not assumed. A 15-feature LightGBM model (7 lags, 3 rolling statistics, 5 calendar features) is trained for regular demand, but the backtest shows Croston's method (with Syntetos-Boylan bias correction) beating it there too (0.444 vs 1.351 lead-time-sum WAPE), so evidence-based routing currently sends regular-demand SKUs to Croston-SBA in production; a conservative recent-mean buffer serves highly intermittent demand. Forecasts feed a reorder-point calculation (lead-time demand + Z-score or residual-based dynamic safety stock) constrained by MOQ/order-multiple/max-order rules, with SKU-specific policy overrides taking precedence over defaults. Every analysis persists an `analysis_runs` row and a linked `prediction_logs` row through SQLAlchemy repositories (deduped within a 15-minute window so repeated dashboard reads don't each write a fresh row); once a prediction's target window completes, it is scored (WAPE/MAE/RMSE/Bias/MASE, daily and lead-time-sum) with explicit leakage guards, rolled into monitoring snapshots that classify recent performance as stable/warning/degraded against a baseline, and can trigger a retraining recommendation. Candidate models are trained and evaluated against an evidence gate (checksum, feature-schema match, minimum test points, horizon compatibility, bias safety, and now beating the strongest reference baseline, not just the active artifact) before a human operator can promote or roll back the serving artifact through a checksum/schema/deserialization-validated, fully audited CLI — automatic retraining and automatic promotion are both explicitly disabled by configuration. Because the dataset is historical and frozen, live monitoring cannot accumulate genuinely new evidence; a separate Historical Monitoring Replay mechanism demonstrates the identical evaluate-monitor-classify pipeline against held-out historical windows, clearly labeled as replay and structurally isolated from the tables live retraining decisions read. A separate inventory-policy simulation (`scripts/compute_kpis.py`, real model + routing, inventory-position tracking, residual-sigma safety stock, two baselines) currently shows the "intelligent" policy roughly at break-even against a naive fixed-threshold baseline in aggregate dollar terms (-1.7%) but clearly worse on a mean-per-SKU basis (-30.5%, 95% CI entirely negative) — an unreconciled divergence, kept as computed rather than tuned to look better. The system is deployed as Next.js (Vercel) + FastAPI (Render) + PostgreSQL (Neon), with Alembic-managed migrations validated in CI against a real PostgreSQL service container alongside 374 backend and 38 frontend tests.

## Key Achievements

- Evidence-driven hybrid forecasting router validated by a corrected, rolling-origin multi-step backtest, not assumption (Croston-SBA measurably beats LightGBM in aggregate *and* on the regular-demand class LightGBM is scoped to — and the system routes accordingly instead of hiding it).
- Found and fixed a real bug in the recursive LightGBM forecast (frozen calendar features, collapsed rolling-std after one step) and in the original one-step-ahead evaluation methodology itself, before trusting any of the numbers above.
- A genuinely controlled MLOps lifecycle: monitoring, degradation detection, retraining recommendation, candidate evaluation, and promotion/rollback — now gated on beating the strongest baseline, not just the active model — all human-gated and fully audited.
- A historical-replay mechanism built specifically to avoid fabricating live monitoring evidence on a frozen dataset — an honest solution to a real constraint rather than a workaround that hides it.
- A real, found-and-fixed concurrency bug in the model-promotion rollback path (a race that could momentarily violate the one-active-artifact database invariant).
- An inventory-policy ROI simulation that reports an unreconciled, non-flattering result (aggregate cost near break-even, mean per-SKU cost clearly worse than naive) rather than smoothing it into a single clean number, because that's what the corrected simulation actually shows.
- Found and fixed a bug in the evaluation script itself: the "production routed" backtest column never actually called the routing service, so it silently reported the legacy default relabeled — a reminder that a pipeline running without errors isn't the same as measuring the right thing.
- 374+ backend tests and 38 frontend tests, with CI validating a full PostgreSQL migration round-trip, not just SQLite.

## Technology List

Python, FastAPI, SQLAlchemy, Pydantic, Alembic, PostgreSQL, LightGBM, pandas, NumPy, scikit-learn, SciPy, PyArrow/Parquet, Next.js, React, TypeScript, GitHub Actions, Render, Vercel, Neon.

## Limitations (state plainly, do not omit)

- Demo dataset is historical and frozen; no live ERP/POS integration exists.
- Live production monitoring has no new evidence to accumulate without a live demand feed — historical replay is a deliberate substitute, not a replacement, and is never presented as live evidence.
- The production LightGBM artifact does not beat Croston-SBA on the regular-demand class it's scoped to, even after retraining with more data and SKU-profile features (0.489 vs 0.444 WAPE, closest candidate) — evidence-based routing currently sends that class to Croston-SBA instead.
- The inventory-policy ROI simulation is roughly break-even against a naive baseline in aggregate dollar terms but clearly worse on a mean-per-SKU basis (95% CI entirely negative) at the default cost assumptions — the divergence itself is not yet explained.
- Forecast-performance monitoring exists; feature/input-distribution drift detection does not.
- Automatic retraining and automatic promotion are both intentionally disabled — every model lifecycle change requires an explicit human command.
- Production assumes a single backend worker; a promotion requires a restart/redeploy to take effect.

This is a decision-support prototype, not autonomous purchasing, a live ERP, a real-time POS system, or a self-healing AI system.
