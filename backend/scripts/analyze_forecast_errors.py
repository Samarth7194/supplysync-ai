"""Turn backtest evidence into the tables that appear in the docs.

Reads ``backend/data/forecast_evaluation_horizons.json`` (produced by
``scripts/evaluate_forecast.py``) and writes:

  * backend/data/forecast_error_analysis.json  - machine-readable rows
  * docs/forecast-evaluation.md                - the human-readable tables

Every number in that markdown file is copied from the evidence JSON; nothing is
typed by hand. Re-run this script after re-running the backtest.

Usage:
    cd backend
    python scripts/analyze_forecast_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

EVIDENCE_PATH = BACKEND_DIR / "data" / "forecast_evaluation_horizons.json"
JSON_OUT = BACKEND_DIR / "data" / "forecast_error_analysis.json"
MARKDOWN_OUT = REPO_DIR / "docs" / "forecast-evaluation.md"
HIGHLY_INTERMITTENT_POLICY_PATH = BACKEND_DIR / "data" / "highly_intermittent_policy_evaluation.json"

REFERENCE = ("predict_zero", "moving_avg_7", "seasonal_naive_7", "croston_sba")
CLASS_ORDER = ("regular", "intermittent", "highly_intermittent", "all")
CLASS_TITLES = {
    "regular": "Regular demand (LightGBM by legacy default; see production routed row for what evidence routing actually selects)",
    "intermittent": "Intermittent demand (Croston-SBA by legacy default)",
    "highly_intermittent": "Highly intermittent demand (conservative buffer by legacy default)",
    "all": "All SKUs pooled",
}
LABELS = {
    "predict_zero": "predict zero",
    "moving_avg_7": "7-day moving average",
    "seasonal_naive_7": "seasonal naive (7)",
    "croston_sba": "Croston-SBA",
    "lightgbm": "LightGBM (production artifact)",
    "production_routed": "production routed (hybrid)",
}


def label(method: str) -> str:
    if method in LABELS:
        return LABELS[method]
    if method.startswith("lightgbm_"):
        return f"LightGBM ({method.removeprefix('lightgbm_')})"
    return method


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:+.0f}%"


def rows_for(class_metrics: dict, comparison: dict | None) -> list[dict]:
    rows = []
    strongest = ((comparison or {}).get("strongest_baseline") or {}).get("method")
    for method, m in class_metrics.items():
        info = ((comparison or {}).get("methods") or {}).get(method) or {}
        diff = info.get("difference_vs_strongest_baseline")
        rows.append({
            "method": method,
            "is_reference": method in REFERENCE,
            "is_strongest_baseline": method == strongest,
            "wape_lead_time_sum": m.get("wape_lead_time_sum"),
            "wape_daily": m.get("wape"),
            "mae": m.get("mae"),
            "bias_ratio": m.get("bias_ratio"),
            "mase": m.get("mase"),
            "n_skus": m.get("n_skus"),
            "n_origins": m.get("n_origins"),
            "beats_strongest_baseline": info.get("beats_strongest_baseline"),
            "difference_vs_strongest_baseline": diff,
        })
    rows.sort(key=lambda r: (r["wape_lead_time_sum"] is None, r["wape_lead_time_sum"] if r["wape_lead_time_sum"] is not None else 0, r["method"]))
    return rows


def build(evidence: dict) -> dict:
    out: dict = {"source": "backend/data/forecast_evaluation_horizons.json", "horizons": {}}
    for horizon, payload in sorted(evidence["horizons"].items(), key=lambda kv: int(kv[0])):
        classes = {}
        for cls in CLASS_ORDER:
            metrics = (payload.get("aggregates") or {}).get(cls)
            if not metrics:
                continue
            classes[cls] = rows_for(metrics, (payload.get("comparisons") or {}).get(cls))
        out["horizons"][horizon] = {
            "period": payload.get("period"),
            "eval_days": payload.get("eval_days"),
            "n_skus": payload.get("n_skus_evaluated"),
            "n_origins": payload.get("n_origins"),
            "routed_method_counts": payload.get("routed_method_counts"),
            "model_artifacts": payload.get("model_artifacts"),
            "classes": classes,
        }
    return out


def markdown(analysis: dict, evidence: dict) -> str:
    lines = [
        "# Forecast evaluation (generated)",
        "",
        "> **Generated file - do not edit by hand.** Produced by "
        "`python scripts/analyze_forecast_errors.py` from `backend/data/forecast_evaluation_horizons.json`, "
        "which `python scripts/evaluate_forecast.py` writes. Re-run both to reproduce every number here.",
        "",
        "## How to read this",
        "",
        "- **Multi-step, rolling-origin.** At every forecast origin the model sees only data up to that day, forecasts the whole horizon at once "
        "(no actuals fed back), and is scored against the demand that really followed.",
        "- **WAPE (lead-time sum)** is the error of the H-day *total* - the quantity a reorder point actually consumes. "
        "**WAPE (daily)** is the error of the individual days.",
        "- **Predict zero always scores WAPE = 1.000.** A method at or above 1.0 is no better than forecasting nothing.",
        "- **Bias** is total forecast minus total actual, as a share of actual demand (+50% = forecast 50% too much).",
        "- **vs strongest baseline** is the paired difference in lead-time-sum WAPE against the best reference method for that class, "
        "with a 95% interval from resampling SKUs (origins inside one SKU are highly correlated, so the SKU is the unit). "
        "A negative number means the model is better; if the interval crosses 0 the result is not distinguishable from a tie.",
        "",
    ]
    for horizon, block in analysis["horizons"].items():
        period = block.get("period") or {}
        lines += [
            f"## Horizon H = {horizon} days",
            "",
            f"{block['n_skus']} SKUs, {block['n_origins']} forecast origins (first origin {period.get('first_origin')}, "
            f"last origin {period.get('last_origin')}; dataset ends {period.get('dataset_end')}; "
            f"models were not trained on data after {period.get('train_cutoff')}).",
            "",
        ]
        if block.get("routed_method_counts"):
            lines += [f"Hybrid routing used: {json.dumps(block['routed_method_counts'], sort_keys=True)}", ""]
        for cls, rows in block["classes"].items():
            n_skus = max((r["n_skus"] or 0) for r in rows) if rows else 0
            lines += [
                f"### {CLASS_TITLES[cls]}",
                "",
                "| Method | WAPE (lead-time sum) | WAPE (daily) | Bias | MASE | Origins | vs strongest baseline (95% CI) |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
            for r in rows:
                diff = r["difference_vs_strongest_baseline"]
                if r["is_strongest_baseline"]:
                    verdict = "strongest baseline"
                elif r["is_reference"]:
                    verdict = "reference"
                elif diff:
                    verdict = f"{diff['point']:+.3f} [{diff['lo']:+.3f}, {diff['hi']:+.3f}]"
                else:
                    verdict = "n/a (too few SKUs)"
                lines.append(
                    f"| {label(r['method'])} | {fmt(r['wape_lead_time_sum'])} | {fmt(r['wape_daily'])} | {pct(r['bias_ratio'])} "
                    f"| {fmt(r['mase'])} | {r['n_origins']} | {verdict} |"
                )
            lines += ["", f"_{n_skus} SKUs in this class at some origin._" if cls != "all" else "", ""]
    if HIGHLY_INTERMITTENT_POLICY_PATH.exists():
        lines += highly_intermittent_policy_section(json.loads(HIGHLY_INTERMITTENT_POLICY_PATH.read_text()))
    lines += [
        "## Caveats",
        "",
        "- One evaluation period (the last 30 days of the dataset, the run-up to Christmas). Results may differ in other seasons.",
        "- SKUs are the highest-volume SKUs by demand before the training cutoff; low-volume SKUs are under-represented.",
        "- Classes are assigned per origin from the trailing 60 days, so a SKU can appear in more than one class.",
        "- The `production routed` row reflects evidence-based routing (ModelRoutingService), gated to the legacy "
        "demand-pattern default whenever evidence is missing, stale, or doesn't clear the improvement/baseline bars.",
        "",
    ]
    return "\n".join(lines)


def highly_intermittent_policy_section(evidence: dict) -> list[str]:
    """WAPE can't select a highly-intermittent policy (predict-zero always
    "wins" WAPE there but never orders), so that class is selected by
    simulated inventory cost instead -- see scripts/evaluate_highly_intermittent_policy.py."""
    lines = [
        "## Highly-intermittent policy selection (by simulated cost, not WAPE)",
        "",
        f"Validation window ending {evidence['validation_window_end']}, "
        f"{evidence['n_skus_simulated']} of {evidence['n_highly_intermittent_skus']} highly-intermittent SKUs simulated "
        f"({evidence['warmup_days']}-day warm-up + {evidence['measured_days']} measured days, "
        f"holding=${evidence['cost_assumptions']['holding_cost_per_unit']}/unit, "
        f"stockout=${evidence['cost_assumptions']['stockout_cost_per_unit']}/unit).",
        "",
        "| Candidate | Total cost | Fill rate |",
        "|---|---:|---:|",
    ]
    for name, c in sorted(evidence["candidates"].items(), key=lambda kv: kv[1]["total_cost"]):
        fill = f"{c['fill_rate']:.1%}" if c["fill_rate"] is not None else "n/a"
        lines.append(f"| {name} | {c['total_cost']:,.0f} | {fill} |")
    lines += [
        "",
        f"**Winner: `{evidence['winner_by_total_cost']}`**"
        + (" (wins at every tested stockout:holding ratio, not just the default 10:1)."
           if evidence["winner_at_every_tested_ratio"] else
           " (ratio-dependent -- see the sensitivity table in the evaluation JSON)."),
        "",
        "Reproduce: `python scripts/evaluate_highly_intermittent_policy.py`",
        "",
    ]
    return lines


def main() -> int:
    if not EVIDENCE_PATH.exists():
        print(f"Missing {EVIDENCE_PATH}. Run: cd backend && python scripts/evaluate_forecast.py")
        return 1
    evidence = json.loads(EVIDENCE_PATH.read_text())
    if evidence.get("evaluation_mode") != "multi_step_rolling_origin":
        print("Evidence file is not from the multi-step backtest; re-run scripts/evaluate_forecast.py.")
        return 1
    analysis = build(evidence)
    JSON_OUT.write_text(json.dumps(analysis, indent=2))
    MARKDOWN_OUT.write_text(markdown(analysis, evidence), encoding="utf-8")
    print(f"Wrote {JSON_OUT.relative_to(REPO_DIR)} and {MARKDOWN_OUT.relative_to(REPO_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
