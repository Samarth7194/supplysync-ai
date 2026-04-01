# SupplySync AI

**Inventory optimization system combining classical supply chain methods with ML forecasting.**

## What This Demonstrates

- **Supply chain domain knowledge**: Correct use of Croston's method for intermittent demand, Z-score safety stock calculations, reorder point formulas, MOQ/order multiple constraints
- **End-to-end ML pipeline**: Raw data (1M+ retail transactions) -> feature engineering -> LightGBM training -> evaluation -> deployed model serving forecasts
- **Simulation-based validation**: 3-policy comparison (naive vs ML vs intelligent) producing real cost/service metrics, not hardcoded numbers
- **Production patterns**: Service layer architecture, centralized model management, structured logging, comprehensive test suite (30 tests)

## Key Results

Measured via simulation on 10 high-volume SKUs from the UCI Online Retail II dataset:

- **37.8% cost reduction** vs naive fixed-threshold policy
- **95.7% fill rate** across simulated SKUs
- **30 passing tests** covering reorder logic, forecasting, constraints, and simulation

## Technical Approach

**Adaptive forecasting** classifies each SKU by demand pattern and applies the best method:
- Regular demand -> LightGBM with 7-day lags, rolling statistics, calendar features
- Intermittent demand -> Croston's method (separate demand size and interval estimation)
- Highly intermittent -> Conservative forecast with safety buffer

**Dynamic safety stock** replaces fixed sigma with rolling forecast error estimation, producing prediction intervals at P80/P90/P95 confidence levels.

**Business constraints** are applied sequentially (MOQ, order multiples, max quantity) with cross-SKU budget optimization using greedy priority allocation.

## Stack

Python, FastAPI, LightGBM, pandas, Next.js, Tailwind CSS

---

*Built by Samarth (2026)*
