# Model candidate comparison (Step 3 — model-quality pass)

> **Source:** the table below is the verbatim output of one command, re-run after
> the recursion fix (Step 2) and after training three new candidates on the
> shared multi-step backtest (`backend/src/evaluation/backtest.py`). It is not
> auto-regenerated like `docs/forecast-evaluation.md` — re-run the two commands
> below to reproduce it exactly.

## How the candidates were trained

Three new LightGBM artifacts, trained on **2,545 SKUs** (every SKU with ≥60
days of recorded demand, not a fixed top-20), **1,299,496** training rows, with
the v2 feature schema (`demand_lag_calendar_sku_v2`: the original 15 lag/
rolling/calendar features plus 3 trailing-28-day SKU demand-profile features —
zero share, mean, coefficient of variation) and an early-stopped train/
validation/test split (30-day test, 14-day validation):

```bash
cd backend
python scripts/train_model.py --objective regression --model-dir saved_models/candidates/regression_v2 \
  --artifact-file lightgbm_regression_v2.pkl --metadata-file lightgbm_regression_v2_metadata.json
python scripts/train_model.py --objective tweedie --model-dir saved_models/candidates/tweedie \
  --artifact-file lightgbm_tweedie.pkl --metadata-file lightgbm_tweedie_metadata.json
python scripts/train_model.py --objective poisson --model-dir saved_models/candidates/poisson \
  --artifact-file lightgbm_poisson.pkl --metadata-file lightgbm_poisson_metadata.json
```

`regression_v2` uses the same L2 objective as the current production model —
it isolates *how much of the improvement is the wider training data and SKU
features alone*, independent of the objective change.

None of these three artifacts have been registered in `model_artifacts` or
promoted. They exist only as local files under `backend/saved_models/candidates/`
(gitignored, like every other `.pkl`) — reproduce them locally with the
commands above.

## How they were evaluated

```bash
python scripts/evaluate_forecast.py --horizons 7,14 --max-skus 200 --no-write \
  --model regression_v2=saved_models/candidates/regression_v2/lightgbm_regression_v2.pkl \
  --model tweedie=saved_models/candidates/tweedie/lightgbm_tweedie.pkl \
  --model poisson=saved_models/candidates/poisson/lightgbm_poisson.pkl
```

Same shared backtest as everything else in this project: rolling-origin,
multi-step, no actuals fed back mid-horizon, 200 SKUs, last 30 days of the
dataset. `lightgbm` is the current production artifact (v1 schema, corrected
recursion from Step 2). `production_routed` is the legacy demand-pattern
policy's actual choice per origin (LightGBM for regular, Croston-SBA for
intermittent, conservative for highly intermittent) — it is *not* a new
method, it is what the live app already does today, included here as the
real-world baseline every candidate needs to beat.

## H = 7 days

### Regular demand

| Method | WAPE (lead-time sum) | Bias | MASE |
|---|---:|---:|---:|
| **Croston-SBA** | **0.444** | -2% | 0.805 |
| lightgbm_tweedie | 0.489 | -22% | 0.699 |
| lightgbm_regression_v2 | 0.522 | +7% | 0.772 |
| lightgbm_poisson | 0.524 | +3% | 0.770 |
| moving_avg_7 | 0.544 | +5% | 0.859 |
| seasonal_naive_7 | 0.544 | +5% | 0.911 |
| predict_zero | 1.000 | -100% | 0.873 |
| production_routed (today's live choice: LightGBM) | 1.351 | +117% | 2.018 |

### Intermittent demand

| Method | WAPE (lead-time sum) | Bias | MASE |
|---|---:|---:|---:|
| **lightgbm_tweedie** | **0.703** | -18% | 1.159 |
| lightgbm_poisson | 0.764 | -6% | 1.208 |
| lightgbm_regression_v2 | 0.793 | -2% | 1.216 |
| moving_avg_7 | 0.879 | -2% | 1.335 |
| seasonal_naive_7 | 0.879 | -2% | 1.452 |
| predict_zero | 1.000 | -100% | 1.061 |
| Croston-SBA / production_routed (today's live choice) | 1.040 | +22% | 1.417 |

### Highly intermittent demand

| Method | WAPE (lead-time sum) | Bias | MASE |
|---|---:|---:|---:|
| **lightgbm_poisson** | **0.988** | -58% | 8.514 |
| predict_zero | 1.000 | -100% | 8.171 |
| lightgbm_tweedie | 1.008 | -66% | 8.446 |
| lightgbm_regression_v2 | 1.189 | -24% | 8.642 |
| moving_avg_7 | 1.227 | -38% | 8.475 |
| production_routed (today's live choice: conservative buffer) | 1.828 | +55% | 8.598 |
| Croston-SBA | 2.067 | +69% | 11.897 |

## H = 14 days

### Regular demand

| Method | WAPE (lead-time sum) | Bias | MASE |
|---|---:|---:|---:|
| **Croston-SBA** | **0.387** | +1% | 0.809 |
| lightgbm_tweedie | 0.424 | -20% | 0.700 |
| lightgbm_poisson | 0.481 | +9% | 0.797 |
| moving_avg_7 / seasonal_naive_7 | 0.506 | +12% | 0.877 / 0.943 |
| lightgbm_regression_v2 | 0.509 | +17% | 0.816 |
| predict_zero | 1.000 | -100% | 0.866 |
| production_routed (today's live choice: LightGBM) | 2.083 | +197% | 2.840 |

### Intermittent demand

| Method | WAPE (lead-time sum) | Bias | MASE |
|---|---:|---:|---:|
| **lightgbm_tweedie** | **0.595** | -24% | 1.335 |
| lightgbm_poisson | 0.639 | -11% | 1.387 |
| lightgbm_regression_v2 | 0.695 | -4% | 1.406 |
| moving_avg_7 / seasonal_naive_7 | 0.737 | -13% | 1.492 / 1.601 |
| Croston-SBA / production_routed (today's live choice) | 0.892 | +11% | 1.574 |
| predict_zero | 1.000 | -100% | 1.250 |

### Highly intermittent demand

| Method | WAPE (lead-time sum) | Bias | MASE |
|---|---:|---:|---:|
| moving_avg_7 / seasonal_naive_7 | 0.932 | -75% | 4.836 / 4.804 |
| **lightgbm_poisson** | **0.942** | -63% | 4.955 |
| lightgbm_tweedie | 0.960 | -73% | 4.873 |
| predict_zero | 1.000 | -100% | 4.535 |
| lightgbm_regression_v2 | 1.138 | -13% | 5.131 |
| production_routed (today's live choice: conservative buffer) | 1.628 | +33% | 4.913 |
| Croston-SBA | 1.932 | +77% | 9.233 |

## What this actually shows

- **The recursion fix (Step 2) plus wider training data and SKU features get
  LightGBM roughly 3x closer to competitive** on the class it's actually
  routed to (regular): production's 1.351 → tweedie's 0.489 at H=7. That's a
  real, large improvement — but it is **not enough to beat Croston-SBA**
  (0.444) on regular-demand SKUs. Regression-v2 (same objective as
  production, just more data + features) also improves a lot (0.522) but not
  as much as tweedie — some of the gain is the objective, some is the data.
- **On intermittent and highly-intermittent SKUs, the new candidates beat
  what the live app actually does today.** `production_routed` sends
  intermittent SKUs to Croston-SBA (1.040) and highly-intermittent SKUs to
  the conservative buffer (1.828) — both worse than `lightgbm_tweedie`
  (0.703) and `lightgbm_poisson` (0.988) respectively. This was not assumed;
  it's what the backtest shows, and it's a genuinely different conclusion
  than the routing policy currently acts on.
- Do not read "highly intermittent" too confidently: `predict_zero` scores
  1.000 there and every method is within shouting distance of it (0.93–2.07)
  — this demand class has very little signal for *any* method, including the
  ones that technically win.
- None of this changes production. These are unregistered, unpromoted
  candidate artifacts evaluated for comparison only.

## What would need to happen for this to reach production

**Update:** Steps 4 and 5 below are now done (see the README's
[What I Found and Changed](../README.md#what-i-found-and-changed)). Evidence-
based routing (`ModelRoutingService`) now consumes this shared multi-step
backtest and requires a method to beat every reference baseline, not just the
legacy default, and the promotion gate (`candidate_evaluation_service.py`)
requires a candidate to beat the strongest baseline, not merely the active
artifact. What's left is the one thing routing/promotion can't fix by
themselves:

1. ~~Evidence-based routing needs to be fed this kind of multi-step evidence
   and given a "must beat the strongest baseline" rule.~~ Done.
2. ~~A candidate needs to clear a promotion gate that requires beating the
   best available method, not just the active artifact.~~ Done.
3. **None of these three candidates has actually been trained on the
   production feature schema and registered.** They still exist only as
   local files under `backend/saved_models/candidates/` (gitignored, never
   registered in `model_artifacts`, never promoted). Even if one were
   registered, none of the three beats Croston-SBA on the regular-demand
   class specifically (best: tweedie at 0.489 vs Croston-SBA's 0.444), so a
   real promotion would only make sense for the intermittent/highly-
   intermittent classes, where the candidates do beat what the live router
   does today for those classes.
