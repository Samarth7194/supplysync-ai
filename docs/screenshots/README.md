# Screenshot capture guide

This folder holds real screenshots of the running app. The root `README.md` references five screenshots — follow this guide to produce all of them in ~10 minutes.

**Ground rules:**
- Real captures only (from a local `make backend` + `make frontend` session). No wireframes, no design mocks, no hand-edited numbers.
- **Don't crop out the `DEMO` pill, the amber synthetic-data banner, or the stock provenance pill.** Those are honesty signals; reviewers should see where values came from.
- Recommended: 1440–1600 px wide, PNG. Retina-scale is fine.

---

## Setup

1. `make backend` (shell 1)
2. `make frontend` (shell 2)
3. Open `http://localhost:3000` in Chrome.
4. Recommended viewport: **1440 × 900**. For mobile variants, use **375 × 812**.
5. Capture: `Cmd/Ctrl + Shift + P` → "Capture full-size screenshot" (whole scrollable page) or "Capture screenshot" (viewport only).

Save each file under `docs/screenshots/` with the exact filename below so the root README's image tags resolve.

---

## 1. `01-dashboard.png` — Dashboard overview

- **URL:** `http://localhost:3000`
- **Capture:** Full-size screenshot.
- **What to show:** Hero, readiness pipeline (5-step flow), all 4 KPI cards, top ~10 rows of the SKU table with risk badges and the new row-level color bars, and the Recent analyses panel beneath.
- **Tip:** Let the page fully load — filter buttons should show counts, and a few "re-analyzing…" spinners in the table are fine (proves the system is live, not mocked).

## 2. `02-sku-ml-path.png` — SKU detail (ML forecast path)

- **URL:** `http://localhost:3000/sku/85099B`
- **Pre-state:** default sliders (7 days / 95%), default demo stock.
- **Capture:** Full-size screenshot.
- **What to show:** Hero recommendation card (coloured to match risk band), the explanation trio ("Why this forecast path"), the historical demand chart with P50/P90 reference lines, the decision rationale panel, and the Model & Method provenance block with `artifact_available: true`.
- **Why this SKU:** `85099B` (Jumbo Bag Red Retrospot) is a top-volume SKU in training — `forecast_source` should be `model_forecast`.

## 3. `03-sku-synthetic-path.png` — SKU detail (fallback path)

- **URL:** `http://localhost:3000/sku/NEVER_SEEN_42` (any SKU not in the dataset)
- **Capture:** Full-size screenshot.
- **What to show:** The amber "Synthetic demo data" banner at the top and the decision block using the statistical / rule-based fallback — proof that unknown SKUs are handled transparently.
- **Tip:** Scroll so both the banner and the decision card are in the frame.

## 4. `04-sliders-rerun.png` — Sliders driving the recommendation

- **URL:** `http://localhost:3000/sku/85099B`
- **Pre-state:** Drag **Lead time** to 21 days and **Service level** to 99%. Wait for the "re-analyzing…" spinner to complete.
- **Capture:** Viewport screenshot (no need for full-page).
- **What to show:** Planning Assumptions card with the two sliders visibly moved (live numeric chips should read `21 days` / `99%`), the recommendation changed (larger order qty, different risk colour), and the decision rationale reflecting the new inputs.

## 5. `05-stock-override.png` — User-editable stock input

- **URL:** `http://localhost:3000/sku/85099B`
- **Pre-state:** Click the **Current stock** input in the Planning Assumptions card. Type a value that flips the risk band (try `5` to force HIGH, or `500` to force LOW). Press Enter or blur so the analysis re-runs.
- **Capture:** Viewport screenshot.
- **What to show:** The Current stock input with the user-entered value, the stock provenance pill, and the hero recommendation reflecting the new stock.

---

## Optional extras

- `06-dashboard-mobile.png` — `375 × 812` viewport of the dashboard, showing the responsive pipeline divider and single-column KPI stack.
- `07-recent-analyses.png` — zoomed-in crop of the Recent analyses panel with several persisted rows.
- `08-cross-sku-cli.png` — terminal screenshot of `python scripts/evaluate_cross_sku.py --folds 5` output. Good supporting image for the bring-your-own-data section.

---

## Optimizing file size

Chrome full-page captures are usually 500 KB – 2 MB. If you want smaller files:

```bash
# macOS
pngquant --quality 65-85 --ext .png --force docs/screenshots/*.png

# Linux
optipng -o5 docs/screenshots/*.png

# Windows (via ImageMagick)
magick mogrify -quality 85 docs/screenshots/*.png
```

Target: under 400 KB per screenshot. No need to edit the README — image tags already point at these filenames.
