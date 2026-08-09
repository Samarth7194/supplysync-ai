"""Simulate evidence-based routing against the offline evaluation artifact.

This script does not train models and does not change production routing. It
answers: if the current routing policy were applied to the committed offline
evaluation evidence, which methods would it select and how would the primary
metric compare with the legacy default policy?

Usage:
    cd backend
    python scripts/simulate_model_routing.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SRC_DIR))

from config.settings import load_settings  # noqa: E402
from services.model_routing_service import ModelRoutingService  # noqa: E402


RUNTIME_TO_OFFLINE = {
    "ml_lightgbm": "lightgbm",
    "croston": "croston_sba",
}


def main() -> int:
    settings = load_settings()
    routing_settings = replace(settings.forecasting, evidence_routing_enabled=True)
    path = BACKEND_DIR / "data" / "forecast_evaluation.json"
    if not path.exists():
        print("Offline evaluation artifact not found: backend/data/forecast_evaluation.json")
        print("Run: cd backend && python scripts/evaluate_forecast.py")
        return 1

    with path.open() as fh:
        artifact = json.load(fh)

    horizon = int(artifact.get("horizon_days") or 0)
    router = ModelRoutingService(
        settings=routing_settings,
        forecast_evaluation_repository=None,
        offline_evaluation_path=path,
    )
    as_of = datetime.now(timezone.utc).date()
    metric = settings.forecasting.routing_primary_metric

    selection_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    default_errors: list[float] = []
    selected_errors: list[float] = []
    default_bias: list[float] = []
    selected_bias: list[float] = []

    rows = artifact.get("per_sku") or []
    for row in rows:
        sku = row.get("sku")
        demand_class = row.get("demand_class")
        metrics = row.get("metrics") or {}
        if not sku or not demand_class:
            continue

        decision = router.select_method(
            sku_code=str(sku),
            demand_pattern=str(demand_class),
            forecast_horizon=horizon,
            as_of_date=as_of,
        )
        selection_counts[decision.selected_method] += 1
        source_counts[decision.selection_source] += 1

        default_offline = RUNTIME_TO_OFFLINE.get(decision.default_method)
        selected_offline = RUNTIME_TO_OFFLINE.get(decision.selected_method)
        if not default_offline or not selected_offline:
            continue
        default_metric = (metrics.get(default_offline) or {}).get(metric)
        selected_metric = (metrics.get(selected_offline) or {}).get(metric)
        if default_metric is None or selected_metric is None:
            continue
        default_errors.append(float(default_metric))
        selected_errors.append(float(selected_metric))

        default_bias_value = (metrics.get(default_offline) or {}).get("bias")
        selected_bias_value = (metrics.get(selected_offline) or {}).get("bias")
        if default_bias_value is not None and selected_bias_value is not None:
            default_bias.append(float(default_bias_value))
            selected_bias.append(float(selected_bias_value))

    evidence_decisions = sum(count for source, count in source_counts.items() if source != "default")
    fallback_decisions = source_counts.get("default", 0)

    print(f"SKUs evaluated: {len(rows)}")
    print(f"Forecast horizon: {horizon}")
    print(f"Primary metric: {metric}")
    print(f"Decisions using evidence: {evidence_decisions}")
    print(f"Decisions using fallback: {fallback_decisions}")
    print(f"Method selection counts: {dict(selection_counts)}")
    print(f"Evidence source counts: {dict(source_counts)}")
    if default_errors and selected_errors:
        print(f"Average {metric} under default policy: {sum(default_errors) / len(default_errors):.4f}")
        print(f"Average {metric} under evidence policy: {sum(selected_errors) / len(selected_errors):.4f}")
    else:
        print("Average error comparison unavailable for the selected/default methods.")
    if default_bias and selected_bias:
        print(f"Average bias under default policy: {sum(default_bias) / len(default_bias):.4f}")
        print(f"Average bias under evidence policy: {sum(selected_bias) / len(selected_bias):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
