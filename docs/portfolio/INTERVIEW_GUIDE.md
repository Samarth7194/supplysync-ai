# Interview Guide — SupplySync AI

Practical prep for discussing this project in a technical interview. Answers are written to be honest under follow-up questions, not just impressive on first pass.

---

## 30-Second Explanation

"SupplySync AI is an ML-powered inventory decision-support tool. It takes a SKU's historical demand, classifies whether that demand is regular, intermittent, or highly intermittent, forecasts it with the method suited to that pattern — LightGBM, Croston-SBA, or a conservative buffer — and converts the forecast into a reorder recommendation using lead-time demand, uncertainty-aware safety stock, and supplier constraints like MOQ. It also has a controlled MLOps layer: it logs predictions, evaluates them once real outcomes exist, monitors for degradation, and supports human-approved model promotion and rollback."

## 90-Second Explanation

Extend the 30-second version with: "The interesting part isn't really the forecasting model — it's that no single forecasting method works well across all demand patterns on real retail data, and that includes the one it's actually scoped to. LightGBM doesn't just lose on sparse, intermittent SKUs — a corrected, multi-step backtest shows Croston-SBA beating it on *regular*-demand SKUs too (0.444 vs 1.351 WAPE). So the system runs an evidence-based router that only switches a class away from its legacy default when a method actually beats every reference baseline in the backtest, and with the evidence currently committed, that sends regular-demand SKUs to Croston-SBA — not LightGBM. I validated that with the same offline backtest before wiring it live, and I didn't stop there: finding that result meant fixing the backtest methodology itself (it used to be a flawed one-step-ahead evaluation) and a real bug in the recursive LightGBM forecast before trusting any of these numbers.

The other half is the MLOps lifecycle. Since the demo runs on a frozen historical dataset with no live ERP feed, I couldn't just let live monitoring sit empty — but I also refused to fabricate live actuals to make a dashboard look populated. So I built a historical replay mechanism that runs the exact same evaluate → monitor → classify pipeline against held-out historical windows, clearly labeled as replay, not live evidence, and structurally unable to trigger retraining or promotion. Promotion and rollback are both real, evidence-gated, audited operations — but they're CLI-only and human-approved by design; nothing auto-trains or auto-promotes."

## Full Technical Walkthrough

1. **Data**: UCI Online Retail II, cleaned and aggregated to ~531K daily-demand rows across ~4,900 SKUs (2009-12-01 → 2011-12-09).
2. **Classification**: zero-demand share > 80% → highly intermittent; > 50% → intermittent; otherwise regular.
3. **Forecasting**: a hybrid router picks LightGBM (15 features: 7 lags, 3 rolling stats, 5 calendar features), Croston-SBA, or a conservative buffer based on the classification, with a moving-average fallback if the ML path is unavailable.
4. **Uncertainty**: rolling forecast-error sigma from logged residuals when there's enough evidence; otherwise, an offline-backtest-derived residual sigma for that demand class; otherwise historical demand standard deviation.
5. **Inventory decision**: lead-time demand + Z-score safety stock = reorder point; compared against current stock, then rounded through MOQ → order multiple → max-order cap.
6. **Persistence**: every analysis writes an `analysis_runs` row and a linked `prediction_logs` row via SQLAlchemy repositories, so every forecast has an audit trail from the moment it's made.
7. **Evaluation**: once a prediction's target window has actually passed, it's scored against real recorded demand (WAPE/MAE/RMSE/Bias/MASE) — never before the window completes.
8. **Monitoring**: evaluated forecasts roll into a monitoring snapshot classified stable/warning/degraded against a baseline, with persistence-based degradation (a single bad window doesn't flip the state).
9. **Retraining recommendation → candidate training → candidate evaluation → promotion**: a fully evidence-gated pipeline, but every step past "recommend" requires an explicit operator command.
10. **Deployment**: Next.js on Vercel, FastAPI on Render, PostgreSQL on Neon, Alembic-migrated schema, CI-validated on every push.

## Why Hybrid Forecasting?

Because a single global model doesn't dominate on real retail demand — and the margin is large. The corrected multi-step, rolling-origin backtest (200 SKUs, H=7) shows Croston-SBA beating LightGBM in aggregate WAPE (0.490 vs 1.560, lead-time-sum), with LightGBM's bias turning sharply positive across every class, worst on intermittent/highly-intermittent SKUs (+456% and +697% respectively). Rather than accept that or force one model everywhere, the system routes per demand pattern based on that evidence, with a hard rule: a method is only selected if it beats every reference baseline, not just the legacy default.

## Why LightGBM?

It's fast to train, handles tabular lag/calendar features well without heavy preprocessing, and gives interpretable feature importances. It was never assumed to be the best choice for every SKU, and the backtest bears that out: on the exact class it's scoped to (regular demand), Croston-SBA still beats it (0.444 vs 1.351 WAPE), even after retraining LightGBM on 2,545 SKUs with SKU-profile features and a tweedie objective (best result: 0.489 — closer, but still short). Evidence-based routing acts on that directly: regular-demand SKUs currently route to Croston-SBA in production, not LightGBM. LightGBM candidates do win on intermittent and highly-intermittent SKUs relative to what the live router does today for those classes — see [docs/model-candidates-comparison.md](../model-candidates-comparison.md) — but none of those candidates are registered or promoted yet.

## Why Croston-SBA?

Croston's method is built for intermittent demand: it separately tracks demand size and inter-arrival interval instead of averaging over zeros, which is what causes naive/moving-average methods to systematically misjudge sparse demand. The SBA correction removes Croston's known positive bias. It's a well-established classical method for this exact problem, not a novel choice — the value here is knowing when to reach for it instead of defaulting to ML.

## How Safety Stock Works

Two modes: **dynamic**, using the rolling standard deviation of recent forecast residuals (actual − predicted) when there's enough logged evidence, and **traditional**, `Z(service_level) × σ × √(lead_time_days)` otherwise. Dynamic safety stock adapts to how wrong the model has actually been recently rather than assuming a fixed distribution.

The traditional formula's σ itself has a fallback order, not just one historical number: logged SKU-method residuals → logged SKU residuals → logged demand-class residuals → an **offline-backtest-derived residual sigma for that demand class** (new: reuses the same sums the shared backtest already keeps for bias/RMSE, so it's not a separate computation) → historical demand standard deviation, only as the last resort. A fresh deployment with zero logged predictions still gets a real forecast-error sigma from the committed backtest instead of falling straight to raw demand volatility, which overstates uncertainty for well-forecastable SKUs and understates it for genuinely erratic ones.

## How Forecasts Become Reorder Decisions

`lead_time_demand = sum(forecast over lead time)` → `reorder_point = lead_time_demand + safety_stock` → `raw_order = max(0, reorder_point − current_stock)` → apply MOQ, then round to the nearest order multiple, then cap at the maximum order quantity if one is configured. Persisted per-SKU policy overrides take precedence over the demand-pattern default when one exists.

## How Evaluation Avoids Leakage

A prediction is only eligible for evaluation once its target window has fully completed *and* real recorded demand exists for that exact window — verified by comparing the prediction's target dates against the bounds of the recorded demand series before scoring. Nothing is ever evaluated against demand that occurred after the forecast was made, and Historical Replay enforces the same rule in the other direction: it truncates history at the anchor date before generating a forecast, so the model never sees the actuals it's about to be scored against. This is covered by explicit tests, including one that mutates only the post-anchor portion of a series and asserts the forecast is unaffected.

## MLOps Lifecycle

Prediction logging → forecast evaluation (once real outcomes exist) → monitoring snapshot → degradation classification → retraining recommendation → candidate training → candidate evaluation → human-approved promotion, with rollback available to any prior valid artifact at any point. Every promotion or rollback is preflight-validated (checksum, feature-schema match, successful deserialization) and written to an audit table (`model_promotion_events`) regardless of outcome.

## Promotion / Rollback Safety

Promotion requires a *completed* `retraining_runs` row whose candidate evaluation is marked promotion-eligible — a candidate is never promotable just because it exists. Both promotion and rollback go through the same preflight (checksum, feature schema, deserialization) before any database state changes, and the one-active-artifact-per-model invariant is enforced by deactivating the current active artifact and flushing *before* activating the new one, specifically to avoid a race where two artifacts are briefly active at once (a real bug found and fixed during development). Rollback never deletes or overwrites artifact files — it only changes which one is marked active.

## Why Automatic Retraining/Promotion Are Disabled

Because retraining and promotion change what's actually making decisions in production, and I wanted a human in the loop before that happens — not because the automation would be hard to build. `AUTO_RETRAIN_ENABLED=false` is explicit in configuration, and there is no code path anywhere that trains or promotes a model without an operator running a CLI command. The system can recommend and can evaluate a candidate as eligible; it stops there by design.

## Historical Replay vs Live Monitoring

Live monitoring needs real predictions **and** the genuine future demand that later arrives for them. The dataset this project runs on is historical and frozen — it ends in December 2011 — so predictions made "today" target windows the dataset can never fill in with real actuals. Rather than fabricate demand or leave the monitoring feature looking permanently broken, Historical Replay runs the identical evaluate → monitor → classify pipeline against **already-recorded** historical demand at an earlier anchor date. It reuses the real forecasting code and real classification thresholds, but it's labeled `historical_replay` everywhere it appears and is architecturally incapable of writing to the same tables live monitoring uses — so it can't accidentally influence a real retraining decision.

## Why Is LightGBM's WAPE So High?

Because of a real, since-fixed bug, plus a genuine limit of the model on this data. The recursive multi-step forecast used to mutate a single static feature row at each step instead of rebuilding lag/rolling/calendar features from the extended series — calendar features froze and `rolling_std` collapsed to zero after the first step. Fixing that cut LightGBM's lead-time-sum WAPE roughly 3x on regular-demand SKUs (3.736 → 1.351 at H=7). That's a real improvement, but it's still worse than every reference baseline, including plain predict-zero (1.000). I'm not tuning past that to make the number look better — a WAPE above 1.0 on a corrected, multi-step backtest is a real result: LightGBM isn't the right tool for this class of demand on this dataset, at least not the plain-regression objective currently in production. The honest response is architectural, not cosmetic: route each SKU to whichever method the backtest actually shows winning for its demand shape — which is exactly why evidence-based routing now sends regular-demand SKUs to Croston-SBA instead — and keep monitoring in place to catch it if a routed method starts underperforming its own baseline.

## Biggest Technical Challenges

- **Temporal leakage**: making sure evaluation, historical replay, and the routing evidence system never let target-window data influence the forecast that's being scored against it. Solved with explicit date-boundary checks and tests that mutate only post-anchor data and assert the forecast doesn't change.
- **Intermittent demand**: getting an honest read on where LightGBM helps vs. hurts required actually running the offline backtest per demand class, not assuming one model is universally better.
- **The one-active-model-artifact invariant**: a database-level partial unique index enforces exactly one active artifact per model name; a promotion/rollback race during recovery from a failed handoff briefly violated it during development — the fix was ordering the deactivate-then-flush before the reactivate-then-flush.
- **Production artifact loading**: making startup resolve a DB-active artifact first, fall back to a locally configured one, and fall back again to a statistical method — without ever crashing the API if the "best" option isn't available.
- **Evaluation evidence for routing**: deciding how much evidence (sample size, recency, horizon match) is enough before letting logged evidence override the safe demand-pattern default, without making routing flap on noise.
- **Historical replay semantics**: designing a mechanism that reuses real production code paths for forecasting and classification without ever touching the live database tables that retraining decisions read from.

## What I Would Improve Next

- Close the remaining regular-demand gap: even the best retrained LightGBM candidate (tweedie objective, wider training data, SKU-profile features) doesn't beat Croston-SBA on that class yet (0.489 vs 0.444 WAPE).
- Investigate why the KPI/ROI simulation's aggregate cost (-1.7% vs. naive, close to break-even) and mean per-SKU savings (-30.5%, 95% CI entirely negative) diverge so sharply on the held-out window — a handful of low-cost SKUs likely swing the mean, but that's not yet confirmed per-SKU.
- Add real feature/input-distribution drift detection alongside the existing forecast-performance monitoring.
- Connect a live ERP/POS feed so live monitoring can accumulate genuine new evidence instead of relying on historical replay.
- Move to probabilistic (quantile or conformal) forecasting instead of the current residual-based uncertainty approximation.
- Support multi-worker runtime synchronization so a promotion doesn't require a full redeploy to take effect everywhere.

## Common Interview Questions

**Why not use LSTM or a deep learning model?**
The dataset is small per SKU (roughly a two-year daily series) and demand is sparse for most SKUs — deep sequence models typically need more data than that to beat well-tuned classical/tree-based baselines, and the offline backtest already shows the production LightGBM artifact isn't the best option on *any* demand class on this data, let alone one a heavier model would fix. Adding model complexity without evidence it helps would be the wrong lesson from this project.

**Why not use one model for all SKUs?**
Because the backtest shows it doesn't work well — LightGBM's bias goes strongly positive on every class, worst on intermittent and highly-intermittent SKUs specifically (+456% and +697% respectively), while Croston-SBA (intermittent's legacy default) and a conservative buffer (highly-intermittent's) are both built for exactly that case. Routing by demand pattern is a direct response to measured evidence, not an assumption — and the same evidence is what now moves the *regular*-demand default away from LightGBM too.

**How do you prevent data leakage?**
Every evaluation path checks that the target window has actually completed and that real recorded demand exists for it before scoring; historical replay explicitly truncates the demand series at the anchor date before generating any forecast. Both are covered by tests.

**Why WAPE instead of MAPE?**
MAPE explodes when actual demand is near zero — extremely common on intermittent SKUs, where a single-unit actual against a two-unit forecast is a "100% error." WAPE (`sum|error| / sum|actual|`) is well-defined as long as total actual demand isn't zero and is far less noisy on sparse data.

**What happens when the model file is corrupted or missing?**
Startup resolves the runtime model in priority order: a valid DB-active artifact, then a configured local artifact, then a statistical fallback. Each candidate is checksum- and feature-schema-validated before being trusted; if none pass, the API still starts and regular-demand forecasts fall back to a simple moving average rather than crashing.

**How does rollback work?**
The operator runs a CLI command with an exact target artifact ID. The service preflight-validates that artifact (checksum, feature schema, deserialization) before touching the database, deactivates the current active artifact, activates the target, and writes an audit event — all inside one transaction, so a failure partway through restores the original state instead of leaving two artifacts active or none.

**Why disable auto promotion?**
Because a model change is a production behavior change, and I'd rather have a human confirm a candidate's evaluation looks right than trust a fully automated loop with no review step — especially at this project's scale, where the evaluation evidence per candidate is still relatively small.

**What is the difference between backtest, historical replay, and live monitoring?**
Offline backtest compares methods on a fixed historical holdout to pick baselines and validate routing. Historical replay runs the live monitoring *pipeline itself* against held-out historical windows to prove it works end-to-end. Live monitoring is the real thing — evaluated production predictions — which currently has no new evidence to work with because there's no live demand feed.

**What would make this truly production-ready?**
A live ERP/POS integration so live monitoring has real ongoing evidence, feature/distribution drift detection (not just performance monitoring), multi-worker-safe promotion without requiring a redeploy, and probabilistic forecasting with calibrated intervals instead of the current residual approximation.

**How would you connect ERP/POS data?**
Add an ingestion adapter that writes into the same `skus`/demand-history shape the rest of the system already reads from `DataService`, so forecasting, evaluation, and monitoring code wouldn't need to change — only the data source underneath `DataService` would.

**Does this place real purchase orders?**
No. It's decision support — it returns a recommended order quantity and the reasoning behind it; a human or a separate procurement system would act on it.

**How is forecast uncertainty computed?**
From the standard deviation of recent forecast residuals (actual − predicted) when there's enough logged evidence for that SKU/method/horizon combination; if not, from that demand class's residual sigma in the offline backtest (reusing sums the backtest already computes, not a new pass); only then does it fall back to the historical demand standard deviation.

**What database do you use and why SQLAlchemy + Alembic?**
PostgreSQL in production (SQLite locally for convenience), with SQLAlchemy repositories so route handlers never touch the ORM directly, and Alembic-managed migrations so schema changes are versioned and CI-validated (including a full upgrade → downgrade → upgrade round-trip against a real Postgres container).

**How do you know the promotion evidence gate actually works?**
It's covered by tests that assert a candidate with no completed retraining run, or one marked ineligible by its own evaluation, is rejected — and a separate test proves the exact zero-active-artifact bootstrap edge case (used once, to activate the very first production model) still enforces the same preflight and invariant.

**What's the single most interesting bug you found while building this?**
A race in the promotion rollback-restore path: recovering from a failed runtime handoff could reactivate the previous artifact before deactivating the target *in the same flush*, momentarily violating the one-active-artifact database constraint. Fixed by explicitly ordering deactivate-then-flush before reactivate-then-flush.

**Did evidence-based routing actually work once you turned it on?**
The routing *logic* did — I verified `ModelRoutingService.select_method()` directly and it correctly picked Croston over LightGBM for regular demand. But the backtest script that reports "what does production actually do" never passed it a routing service at all, so every "production routed" number in every table was silently the legacy default relabeled, not a real measurement. The fix was a one-line constructor argument; finding it required distrusting a friendly-looking number (a WAPE that happened to equal LightGBM's exactly, for every regular-class origin) rather than assuming the pipeline that reported it was correct just because it ran without errors.

**Why did fixing three real bugs make the KPI simulation's per-SKU result worse?**
It didn't uniformly get worse — the aggregate dollar cost improved (-9.0% → -1.7% vs. naive) and fill rate improved (86.8% → 90.5%), because regular-demand SKUs now get better forecasts and highly-intermittent SKUs get a cheaper buffer. But the *mean per-SKU* percentage swung the other way (-13.0% → -30.5%, with the 95% CI now entirely negative). That's a real, currently-unexplained divergence between a ratio-of-sums metric and a mean-of-ratios metric, most likely driven by a handful of low-absolute-cost SKUs swinging heavily in percentage terms — reported as found, not smoothed over, and flagged as follow-up work.
