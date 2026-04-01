# SupplySync AI

**ML-powered inventory decision system with adaptive forecasting and simulation-validated optimization.**

Built on the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (1M+ transactions, 4,900 SKUs).

## What It Does

Takes raw sales data and produces actionable reorder recommendations:

1. **Classifies** each SKU's demand pattern (regular, intermittent, highly intermittent)
2. **Forecasts** demand using the appropriate method per pattern
3. **Computes** safety stock and reorder points with uncertainty quantification
4. **Applies** business constraints (MOQ, order multiples, budget limits)
5. **Explains** every decision with a full calculation breakdown

## Performance (Measured on Real Data)

Computed via simulation on 10 high-volume SKUs from the Online Retail II dataset:

| Metric | Value |
|--------|-------|
| Cost Savings vs Naive | **37.8%** |
| Average Fill Rate | **95.7%** |
| SKUs Analyzed | 10 |
| Model MAE | 95.62 units |

*Run `python scripts/compute_kpis.py` to reproduce these numbers.*

## Architecture

```
Online Retail II CSV
       |
  Data Pipeline (load_retail_data.py)
       |
  Feature Engineering (lag_features.py, time_features.py)
       |
  LightGBM Training (train_model.py)
       |
  Adaptive Forecasting Service
  - Regular SKUs --> LightGBM with lag/calendar features
  - Intermittent  --> Croston's method
  - Highly sparse --> Conservative forecast (1.5x buffer)
       |
  Reorder Decision Engine
  - Dynamic safety stock (rolling forecast error)
  - Prediction intervals (P80/P90/P95)
  - Risk-aware modes (conservative/moderate/aggressive)
       |
  Business Constraints
  - Minimum order quantities
  - Order multiples
  - Budget optimization across SKUs
       |
  FastAPI + Next.js Frontend
```

## Tech Stack

- **ML**: LightGBM, scikit-learn, pandas, numpy
- **API**: FastAPI, uvicorn, Pydantic
- **Frontend**: Next.js 15, React, Tailwind CSS
- **Simulation**: Custom day-by-day inventory simulator with 3-policy comparison
- **Data**: Online Retail II dataset (UCI), Parquet storage

## Quick Start

```bash
# 1. Setup
cd backend
pip install -r requirements.txt

# 2. Train model on real data
python scripts/train_model.py

# 3. Compute KPIs from simulation
python scripts/compute_kpis.py

# 4. Run tests
python -m pytest tests/ -v

# 5. Start API
uvicorn main:app --port 8000

# 6. Start frontend (separate terminal)
cd frontend
npm install && npm run dev
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Analyze SKU risk and get reorder recommendation |
| GET | `/api/kpis` | System-level KPIs from simulation |
| GET | `/health` | Server status |
| POST | `/forecast` | Demand forecast for a SKU |
| POST | `/reorder` | Full reorder decision with constraints |

## Tests

```bash
cd backend
python -m pytest tests/ -v
# 30 tests covering: reorder logic, business constraints,
# forecasting methods, and simulation engine
```

## Project Structure

```
supplysync-ai/
├── backend/
│   ├── src/
│   │   ├── services/          # Core business logic
│   │   ├── inventory/         # Reorder point, constraints
│   │   ├── forecasting/       # LightGBM, Croston's, forecast service
│   │   ├── uncertainty/       # Safety stock, prediction intervals
│   │   ├── simulation/        # Day-by-day simulator, policy comparison
│   │   ├── features/          # Lag and calendar features
│   │   ├── ingestion/         # Data loading and cleaning
│   │   └── api/               # FastAPI routes and schemas
│   ├── scripts/               # Training and KPI computation
│   ├── saved_models/          # Trained LightGBM model
│   └── tests/                 # 30 tests with assertions
├── frontend/                  # Next.js dashboard
├── data/
│   ├── raw/                   # Online Retail II CSV
│   └── processed/             # Daily demand parquet
└── docker-compose.yml
```

## What's Implemented vs Planned

**Implemented and working:**
- Data pipeline: CSV cleaning, aggregation, feature engineering
- LightGBM model trained on real data with temporal split
- Adaptive forecasting by demand pattern (3 methods)
- Dynamic safety stock with rolling forecast error
- Prediction intervals (P80/P90/P95)
- Business constraints (MOQ, multiples, budget optimization)
- Decision explainability service
- 3-policy simulation comparison
- FastAPI backend with real data integration
- Next.js frontend dashboard
- 30 passing tests

**Aspirational (not yet functional):**
- TFT (Temporal Fusion Transformer) integration
- GNN supply chain ripple analysis
- Multi-echelon inventory optimization
- Supplier performance modeling
- Real-time demand sensing

## License

MIT
