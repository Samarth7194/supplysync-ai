"""Tests for the scripts that turn backtest runs into committed evidence and docs."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

from evaluation import backtest as bt  # noqa: E402

evaluate_forecast = importlib.import_module("evaluate_forecast")
analyze = importlib.import_module("analyze_forecast_errors")


def _series(seed: int, n: int = 140) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.poisson(9, n).astype(float)
    return pd.Series(values, index=pd.date_range("2021-01-01", periods=n, freq="D"))


def _evidence(horizon: int = 7) -> dict:
    def biased_high(sku, history, h):
        return bt.Forecast((100.0,) * h, "biased_high")

    result = bt.run_backtest(
        {f"S{i}": _series(i) for i in range(10)},
        {"lightgbm": biased_high},
        bt.BacktestConfig(horizon=horizon),
    )
    payload = result.to_payload(extra={"comparisons": evaluate_forecast.compare_to_strongest_baseline(result, n_boot=40)})
    return {"schema_version": 2, "evaluation_mode": bt.EVALUATION_MODE, "horizons": {str(horizon): payload}}


def test_comparison_names_strongest_baseline_and_flags_a_model_that_loses():
    result = bt.run_backtest(
        {f"S{i}": _series(i) for i in range(10)},
        {"lightgbm": lambda sku, history, h: bt.Forecast((100.0,) * h, "lightgbm")},
        bt.BacktestConfig(horizon=7),
    )
    comparison = evaluate_forecast.compare_to_strongest_baseline(result, n_boot=40)

    regular = comparison["regular"]
    assert regular["strongest_baseline"]["method"] in bt.REFERENCE_METHODS
    assert regular["methods"]["lightgbm"]["beats_strongest_baseline"] is False
    diff = regular["methods"]["lightgbm"]["difference_vs_strongest_baseline"]
    assert diff["point"] > 0  # worse than the strongest baseline
    assert diff["lo"] <= diff["point"] <= diff["hi"]


def test_report_tables_are_generated_from_the_evidence_not_typed_by_hand():
    evidence = _evidence()
    analysis = analyze.build(evidence)
    text = analyze.markdown(analysis, evidence)

    assert "Generated file - do not edit by hand" in text
    assert "predict zero" in text and "| 1.000 |" in text  # predict-zero always scores 1.000
    assert "Horizon H = 7 days" in text
    regular_rows = analysis["horizons"]["7"]["classes"]["regular"]
    assert any(r["method"] == "lightgbm" for r in regular_rows)
    for row in regular_rows:
        formatted = analyze.fmt(row["wape_lead_time_sum"])
        assert formatted in text  # every reported number appears verbatim in the markdown
    assert "nan" not in text.lower()


def test_report_generation_refuses_legacy_one_step_evidence(tmp_path, monkeypatch):
    legacy = tmp_path / "forecast_evaluation_horizons.json"
    legacy.write_text('{"horizons": {"7": {"aggregates": {}}}}')
    monkeypatch.setattr(analyze, "EVIDENCE_PATH", legacy)
    monkeypatch.setattr(analyze, "JSON_OUT", tmp_path / "out.json")
    monkeypatch.setattr(analyze, "MARKDOWN_OUT", tmp_path / "out.md")

    assert analyze.main() == 1
    assert not (tmp_path / "out.md").exists()


def test_committed_evidence_is_multi_step_and_free_of_private_paths():
    """Guards the committed evidence files against regressing to the old one-step format."""
    import json

    for name in ("forecast_evaluation.json", "forecast_evaluation_horizons.json"):
        path = BACKEND_DIR / "data" / name
        text = path.read_text()
        payload = json.loads(text)
        assert payload.get("evaluation_mode") == bt.EVALUATION_MODE, name
        assert payload.get("schema_version") == bt.SCHEMA_VERSION, name
        assert "C:\\\\Users" not in text and "/Users/" not in text and "/home/" not in text, name
