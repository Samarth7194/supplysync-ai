"""Summarize offline forecast errors into a reviewer-friendly artifact."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_payloads() -> list[dict]:
    multi_path = BACKEND_DIR / "data" / "forecast_evaluation_horizons.json"
    legacy_path = BACKEND_DIR / "data" / "forecast_evaluation.json"
    payloads: list[dict] = []
    if multi_path.exists():
        multi = json.loads(multi_path.read_text())
        payloads.extend((multi.get("horizons") or {}).values())
    elif legacy_path.exists():
        payloads.append(json.loads(legacy_path.read_text()))
    return payloads


def _best_model(metrics: dict) -> str | None:
    candidates = {
        model: values.get("wape")
        for model, values in metrics.items()
        if values.get("wape") is not None
    }
    if not candidates:
        return None
    return min(candidates, key=candidates.get)


def _distribution(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(values[0], 4),
        "median": round(median(values), 4),
        "max": round(values[-1], 4),
    }


def build_report() -> dict:
    payloads = _load_payloads()
    method_wins = Counter()
    horizon_summary = {}
    pattern_summary = defaultdict(dict)
    worst_skus = []
    best_skus = []
    wape_by_method = defaultdict(list)
    mae_by_method = defaultdict(list)
    bias_by_method = defaultdict(list)

    for payload in payloads:
        horizon = int(payload.get("horizon_days") or 0)
        aggregates = payload.get("aggregates") or {}
        horizon_summary[str(horizon)] = aggregates.get("all", {})
        for pattern, by_model in aggregates.items():
            if pattern != "all":
                pattern_summary[pattern][str(horizon)] = by_model
            winner = _best_model(by_model)
            if winner:
                method_wins[(str(horizon), pattern, winner)] += 1

        for row in payload.get("per_sku") or []:
            winner = _best_model(row.get("metrics") or {})
            if winner:
                method_wins[(str(horizon), "sku", winner)] += 1
            for model, values in (row.get("metrics") or {}).items():
                if values.get("wape") is not None:
                    wape_by_method[model].append(float(values["wape"]))
                if values.get("mae") is not None:
                    mae_by_method[model].append(float(values["mae"]))
                if values.get("bias") is not None:
                    bias_by_method[model].append(float(values["bias"]))
            lightgbm = (row.get("metrics") or {}).get("lightgbm")
            if lightgbm and lightgbm.get("wape") is not None:
                item = {
                    "horizon_days": horizon,
                    "sku": row.get("sku"),
                    "demand_class": row.get("demand_class"),
                    "lightgbm_wape": lightgbm.get("wape"),
                    "best_model": winner,
                    "total_test_demand": row.get("total_test_demand"),
                    "n_test": row.get("n_test"),
                }
                worst_skus.append(item)
                best_skus.append(item)

    worst_skus = sorted(worst_skus, key=lambda row: row["lightgbm_wape"], reverse=True)[:10]
    best_skus = sorted(best_skus, key=lambda row: row["lightgbm_wape"])[:10]

    return {
        "generated_at": datetime.now().isoformat(),
        "source": "offline_backtest",
        "note": "Offline historical backtest artifact; not live production performance.",
        "horizon_summary": horizon_summary,
        "pattern_summary": dict(pattern_summary),
        "method_win_counts": [
            {"horizon_days": h, "scope": scope, "model": model, "wins": count}
            for (h, scope, model), count in sorted(method_wins.items())
        ],
        "distributions": {
            model: {
                "wape": _distribution(wape_by_method[model]),
                "mae": _distribution(mae_by_method[model]),
                "bias": _distribution(bias_by_method[model]),
            }
            for model in sorted(wape_by_method)
        },
        "worst_lightgbm_skus_by_wape": worst_skus,
        "best_lightgbm_skus_by_wape": best_skus,
    }


def _write_csv(path: Path, report: dict) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "metric", "count", "min", "median", "max"])
        for model, metrics in report["distributions"].items():
            for metric, dist in metrics.items():
                writer.writerow([model, metric, dist.get("count"), dist.get("min"), dist.get("median"), dist.get("max")])


def main() -> int:
    report = build_report()
    out_json = BACKEND_DIR / "data" / "forecast_error_analysis.json"
    out_csv = BACKEND_DIR / "data" / "forecast_error_analysis.csv"
    out_json.write_text(json.dumps(report, indent=2))
    _write_csv(out_csv, report)
    print(f"Saved:\n  {out_json}\n  {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
