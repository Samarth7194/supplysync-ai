# Forecast evaluation (generated)

> **Generated file - do not edit by hand.** Produced by `python scripts/analyze_forecast_errors.py` from `backend/data/forecast_evaluation_horizons.json`, which `python scripts/evaluate_forecast.py` writes. Re-run both to reproduce every number here.

## How to read this

- **Multi-step, rolling-origin.** At every forecast origin the model sees only data up to that day, forecasts the whole horizon at once (no actuals fed back), and is scored against the demand that really followed.
- **WAPE (lead-time sum)** is the error of the H-day *total* - the quantity a reorder point actually consumes. **WAPE (daily)** is the error of the individual days.
- **Predict zero always scores WAPE = 1.000.** A method at or above 1.0 is no better than forecasting nothing.
- **Bias** is total forecast minus total actual, as a share of actual demand (+50% = forecast 50% too much).
- **vs strongest baseline** is the paired difference in lead-time-sum WAPE against the best reference method for that class, with a 95% interval from resampling SKUs (origins inside one SKU are highly correlated, so the SKU is the unit). A negative number means the model is better; if the interval crosses 0 the result is not distinguishable from a tie.

## Horizon H = 7 days

200 SKUs, 4800 forecast origins (first origin 2011-11-09, last origin 2011-12-02; dataset ends 2011-12-09; models were not trained on data after 2011-11-09).

Hybrid routing used: {"conservative": 272, "croston": 447, "ml_lightgbm": 4081}

### Regular demand (served by LightGBM in the legacy routing policy)

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| Croston-SBA | 0.444 | 0.893 | -2% | 0.805 | 4081 | reference |
| 7-day moving average | 0.544 | 0.950 | +5% | 0.859 | 4081 | reference |
| seasonal naive (7) | 0.544 | 0.992 | +5% | 0.911 | 4081 | reference |
| predict zero | 1.000 | 1.000 | -100% | 0.873 | 4081 | reference |
| LightGBM (production artifact) | 3.736 | 4.075 | +359% | 5.345 | 4081 | n/a (too few SKUs) |
| production routed (hybrid) | 3.736 | 4.075 | +359% | 5.345 | 4081 | n/a (too few SKUs) |

_173 SKUs in this class at some origin._

### Intermittent demand (served by Croston-SBA)

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| 7-day moving average | 0.879 | 1.379 | -2% | 1.335 | 447 | reference |
| seasonal naive (7) | 0.879 | 1.563 | -2% | 1.452 | 447 | reference |
| predict zero | 1.000 | 1.000 | -100% | 1.061 | 447 | reference |
| Croston-SBA | 1.040 | 1.527 | +22% | 1.417 | 447 | reference |
| production routed (hybrid) | 1.040 | 1.527 | +22% | 1.417 | 447 | n/a (too few SKUs) |
| LightGBM (production artifact) | 11.478 | 11.919 | +1135% | 11.177 | 447 | n/a (too few SKUs) |

_28 SKUs in this class at some origin._

### Highly intermittent demand (served by the conservative buffer)

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| predict zero | 1.000 | 1.000 | -100% | 8.171 | 272 | reference |
| 7-day moving average | 1.227 | 1.322 | -38% | 8.475 | 272 | reference |
| seasonal naive (7) | 1.227 | 1.467 | -38% | 8.596 | 272 | reference |
| production routed (hybrid) | 1.828 | 2.216 | +55% | 8.598 | 272 | n/a (too few SKUs) |
| Croston-SBA | 2.067 | 2.437 | +69% | 11.897 | 272 | reference |
| LightGBM (production artifact) | 16.226 | 17.016 | +1594% | 75.527 | 272 | n/a (too few SKUs) |

_13 SKUs in this class at some origin._

### All SKUs pooled

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| Croston-SBA | 0.490 | 0.939 | -0% | 1.290 | 4800 | reference |
| 7-day moving average | 0.566 | 0.971 | +4% | 1.197 | 4800 | reference |
| seasonal naive (7) | 0.566 | 1.019 | +4% | 1.258 | 4800 | reference |
| predict zero | 1.000 | 1.000 | -100% | 1.171 | 4800 | reference |
| production routed (hybrid) | 3.611 | 3.956 | +342% | 5.097 | 4800 | n/a (too few SKUs) |
| LightGBM (production artifact) | 4.203 | 4.552 | +405% | 8.596 | 4800 | n/a (too few SKUs) |



## Horizon H = 14 days

200 SKUs, 3400 forecast origins (first origin 2011-11-09, last origin 2011-11-25; dataset ends 2011-12-09; models were not trained on data after 2011-11-09).

Hybrid routing used: {"conservative": 194, "croston": 311, "ml_lightgbm": 2895}

### Regular demand (served by LightGBM in the legacy routing policy)

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| Croston-SBA | 0.387 | 0.910 | +1% | 0.809 | 2895 | reference |
| 7-day moving average | 0.506 | 0.984 | +12% | 0.877 | 2895 | reference |
| seasonal naive (7) | 0.506 | 1.051 | +12% | 0.943 | 2895 | reference |
| predict zero | 1.000 | 1.000 | -100% | 0.866 | 2895 | reference |
| LightGBM (production artifact) | 5.117 | 5.416 | +494% | 7.103 | 2895 | n/a (too few SKUs) |
| production routed (hybrid) | 5.117 | 5.416 | +494% | 7.103 | 2895 | n/a (too few SKUs) |

_173 SKUs in this class at some origin._

### Intermittent demand (served by Croston-SBA)

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| 7-day moving average | 0.737 | 1.287 | -13% | 1.492 | 311 | reference |
| seasonal naive (7) | 0.737 | 1.438 | -13% | 1.601 | 311 | reference |
| Croston-SBA | 0.892 | 1.457 | +11% | 1.574 | 311 | reference |
| production routed (hybrid) | 0.892 | 1.457 | +11% | 1.574 | 311 | n/a (too few SKUs) |
| predict zero | 1.000 | 1.000 | -100% | 1.250 | 311 | reference |
| LightGBM (production artifact) | 14.429 | 14.790 | +1431% | 16.025 | 311 | n/a (too few SKUs) |

_26 SKUs in this class at some origin._

### Highly intermittent demand (served by the conservative buffer)

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| 7-day moving average | 0.932 | 1.114 | -75% | 4.836 | 194 | reference |
| seasonal naive (7) | 0.932 | 1.104 | -75% | 4.804 | 194 | reference |
| predict zero | 1.000 | 1.000 | -100% | 4.535 | 194 | reference |
| production routed (hybrid) | 1.628 | 2.061 | +33% | 4.913 | 194 | n/a (too few SKUs) |
| Croston-SBA | 1.932 | 2.517 | +77% | 9.233 | 194 | reference |
| LightGBM (production artifact) | 19.487 | 20.169 | +1918% | 104.101 | 194 | n/a (too few SKUs) |

_12 SKUs in this class at some origin._

### All SKUs pooled

| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| Croston-SBA | 0.433 | 0.959 | +3% | 1.228 | 3400 | reference |
| 7-day moving average | 0.522 | 0.998 | +10% | 1.098 | 3400 | reference |
| seasonal naive (7) | 0.522 | 1.067 | +10% | 1.163 | 3400 | reference |
| predict zero | 1.000 | 1.000 | -100% | 1.053 | 3400 | reference |
| production routed (hybrid) | 4.896 | 5.209 | +468% | 6.499 | 3400 | n/a (too few SKUs) |
| LightGBM (production artifact) | 5.718 | 6.027 | +554% | 11.936 | 3400 | n/a (too few SKUs) |



## Caveats

- One evaluation period (the last 30 days of the dataset, the run-up to Christmas). Results may differ in other seasons.
- SKUs are the highest-volume SKUs by demand before the training cutoff; low-volume SKUs are under-represented.
- Classes are assigned per origin from the trailing 60 days, so a SKU can appear in more than one class.
- The `production routed` row uses the legacy demand-pattern policy, not evidence routing.
