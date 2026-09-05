# Bring your own data

The headline demo runs on the UCI Online Retail II dataset, but the training, evaluation, and forecasting logic don't depend on that specific CSV. This guide shows three ways to point the pipeline at your own data.

> **Honesty guard:** the cost-savings/fill-rate KPIs served at `GET /api/kpis` (`backend/data/cached_kpis.json`) are specific to Online Retail II's top-10 SKUs. Your numbers will differ. For a fair read on whether this approach fits your data, run the cross-SKU evaluation (§3 below) — it's the honest generalization test.

---

## What the pipeline needs

At its core, the system needs three columns per row:

| Canonical name | What it means | UCI source column |
|---|---|---|
| `date` | Transaction date (parseable by pandas) | `InvoiceDate` |
| `quantity` | Units sold (integers or floats; negative = returns and are dropped) | `Quantity` |
| `sku` | A stable product identifier | `StockCode` |

Everything else (`description`, `price`, `customer`, `country`, `invoice`) is optional. If your CSV already uses these canonical column names, the pipeline works with no configuration.

---

## 1. Same schema as UCI

If your CSV uses `InvoiceDate`, `Quantity`, `StockCode`, etc., just point the pipeline at it:

```bash
# Train on a different dataset
DATA_CSV_PATH=/path/to/my_sales.csv python backend/scripts/train_model.py

# Evaluate baselines on it
DATA_CSV_PATH=/path/to/my_sales.csv python backend/scripts/evaluate_forecast.py
```

Or set it once in `backend/.env`:

```bash
DATA_CSV_PATH=/path/to/my_sales.csv
```

Then run `make bootstrap` as usual.

---

## 2. Different column names

If your CSV uses different column names, provide a JSON mapping. The keys are the **canonical** names the pipeline uses internally; the values are the **source** column names in your CSV.

Create `column_mapping.json`:

```json
{
  "date": "transaction_date",
  "quantity": "units_sold",
  "sku": "product_id",
  "description": "product_name",
  "price": "unit_price"
}
```

Then:

```bash
python backend/scripts/evaluate_custom_dataset.py \
  --csv /path/to/my_sales.csv \
  --column-mapping column_mapping.json \
  --output backend/data/my_evaluation.json
```

Only the keys you care about need to be in the mapping — anything you omit defaults to the UCI column name.

---

## 3. Unseen-SKU generalization check

This is the script that closes the biggest reviewer question: *"Does the model generalize to SKUs it never trained on?"*

```bash
python backend/scripts/evaluate_cross_sku.py --folds 5 --top-skus 20
```

Procedure:

1. Sorts the top-20 SKUs by total demand.
2. Runs 5 folds — each fold holds out 4 SKUs and retrains LightGBM on the remaining 16.
3. Evaluates LightGBM and the four baselines (naive-last, seasonal-naive-7, moving-avg-7, Croston SBA) on each fold's held-out SKUs.
4. Writes `backend/data/cross_sku_evaluation.json` and prints an aggregate summary.

The output looks like this:

```
=== Cross-SKU generalization (5 folds, 20 SKUs) ===

Method                MAE      RMSE     WAPE     Winner rate
lightgbm              ...      ...      ...      ...
naive_last            ...      ...      ...      ...
seasonal_naive_7      ...      ...      ...      ...
moving_avg_7          ...      ...      ...      ...
croston_sba           ...      ...      ...      ...
```

### How to interpret

- **LightGBM wins on unseen SKUs** → the trained model is picking up cross-SKU patterns (calendar effects, lag structure), not just memorizing specific products. This is the strong story.
- **Statistical baselines win on unseen SKUs** → the ML gains are coming from per-SKU idiosyncrasies. That's also useful to know: it suggests Croston / MA are a better *deployment* choice for new products, and the router in [`adaptive_forecasting_service.py`](../backend/src/services/adaptive_forecasting_service.py) is doing the right thing by only sending regular-demand series to LightGBM.

Either outcome is publishable. The cross-SKU JSON ships with the repo.

---

## Zero-shot on a different dataset

If you want to apply the **currently-trained** model to an entirely different dataset without retraining:

```bash
python backend/scripts/evaluate_custom_dataset.py \
  --csv /path/to/another_retailer.csv \
  --column-mapping column_mapping.json
```

The script will print a clear warning at the top of its report:

> This model was trained on UCI Online Retail II top-20 SKUs. Applying it to another dataset without retraining is a zero-shot test. For a fair comparison, retrain via `DATA_CSV_PATH=… python scripts/train_model.py` first.

This is useful for a quick sanity check — are the baselines even computable on this CSV? — but the honest comparison number comes after retraining.

---

## Known limits

- **Daily granularity only.** If your CSV is sub-daily or weekly, you'll need to aggregate it upstream. The pipeline assumes one row per SKU per day after cleaning.
- **No multi-location support.** If your data has `store_id` or `warehouse`, aggregate first or pre-filter.
- **Negative quantities are treated as returns and dropped** during cleaning. If you want to net them, pre-process before passing the CSV.
- **Top-20 SKUs by total demand** is hardcoded as the training subset. Edit `train_model.py` if you want a different cut — it's ~3 lines of change.
- **Column mapping doesn't rename output files.** The model artifact is still written to `saved_models/lightgbm_demand_forecast.pkl` regardless of dataset. Rename manually if you're juggling multiple datasets.
