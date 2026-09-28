"""Rolling-origin, multi-step backtest that runs the production forecast path.

Why this module exists
----------------------
Earlier evaluation code walked a holdout window one day at a time and fed the
*true* previous-day actual back into the model at every step. That measures
one-step-ahead skill, not the multi-day forecast the inventory decision
consumes, and it made "horizon" mean nothing more than "holdout length".

This module replaces those loops. At every forecast origin ``t`` it:

  1. slices the series to ``series.iloc[:t]`` (nothing after the origin),
  2. takes the same trailing window the live analysis path uses
     (``PRODUCTION_HISTORY_DAYS``),
  3. asks every forecaster for ``H`` days in one shot (no actuals fed back),
  4. scores the forecast against the real ``H`` days that followed.

Reported per demand class and per method: daily WAPE, lead-time-sum WAPE (the
error of the ``H``-day total, which is what a reorder point actually uses),
MAE, RMSE, bias, MASE. Reference rows (predict-zero, 7-day moving average,
seasonal naive, Croston-SBA) are always included so a model is never judged in
isolation. Note that a WAPE of 1.0 is exactly what predict-zero scores.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from features.inference_features import build_inference_features
from forecasting.forecast_service import forecast_next_days
from services.adaptive_forecasting_service import (
    PRODUCTION_HISTORY_DAYS,
    adaptive_forecast,
    classify_sku_demand_pattern,
    croston_forecast,
    simple_average_forecast,
)

SCHEMA_VERSION = 2
EVALUATION_MODE = "multi_step_rolling_origin"

REFERENCE_METHODS = ("predict_zero", "moving_avg_7", "seasonal_naive_7", "croston_sba")
PRODUCTION_ROUTED = "production_routed"
LIGHTGBM = "lightgbm"

DEMAND_CLASSES = ("regular", "intermittent", "highly_intermittent")


@dataclass(frozen=True)
class Forecast:
    values: tuple[float, ...]
    method: str | None = None


# A forecaster gets (sku, history, horizon) where ``history`` is already the
# production-length trailing window ending at the origin. It returns None when
# it cannot produce a forecast (the origin is then skipped for *every* method
# so all rows are scored on exactly the same origins).
Forecaster = Callable[[str, pd.Series, int], "Forecast | None"]


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int
    eval_days: int = 30
    stride: int = 1
    min_history_days: int = 30
    history_window: int = PRODUCTION_HISTORY_DAYS

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.eval_days < self.horizon:
            raise ValueError("eval_days must be >= horizon so at least one origin fits")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")


# --------------------------------------------------------------------------
# Reference forecasters (all multi-step, none see data after the origin)
# --------------------------------------------------------------------------

def predict_zero_forecast(sku: str, history: pd.Series, horizon: int) -> Forecast:
    return Forecast((0.0,) * horizon, "predict_zero")


def moving_avg_7_forecast(sku: str, history: pd.Series, horizon: int) -> Forecast:
    return Forecast(tuple(simple_average_forecast(history, horizon)), "moving_avg_7")


def seasonal_naive_7_forecast(sku: str, history: pd.Series, horizon: int) -> Forecast:
    """Repeat the last observed week: day k of the horizon = same weekday last week."""
    values = history.to_numpy(dtype=float)
    if values.size == 0:
        return Forecast((0.0,) * horizon, "seasonal_naive_7")
    if values.size < 7:
        return Forecast((float(values[-1]),) * horizon, "seasonal_naive_7")
    last_week = values[-7:]
    return Forecast(tuple(float(last_week[k % 7]) for k in range(horizon)), "seasonal_naive_7")


def croston_sba_forecast(sku: str, history: pd.Series, horizon: int) -> Forecast:
    return Forecast(tuple(croston_forecast(history, horizon)), "croston_sba")


def reference_forecasters() -> dict[str, Forecaster]:
    return {
        "predict_zero": predict_zero_forecast,
        "moving_avg_7": moving_avg_7_forecast,
        "seasonal_naive_7": seasonal_naive_7_forecast,
        "croston_sba": croston_sba_forecast,
    }


def lightgbm_forecaster(
    model: Any,
    feature_columns: Sequence[str],
    *,
    recursion: Callable[..., list[float]] = forecast_next_days,
) -> Forecaster:
    """Forced-LightGBM forecaster using the production feature + recursion code."""

    columns = list(feature_columns)

    def _forecast(sku: str, history: pd.Series, horizon: int) -> Forecast | None:
        features = build_inference_features(history, columns)
        if features is None:
            return None
        return Forecast(tuple(float(v) for v in recursion(model, features, horizon)), LIGHTGBM)

    return _forecast


def production_forecaster(
    model: Any,
    feature_columns: Sequence[str] | None,
    *,
    routing_service: Any | None = None,
) -> Forecaster:
    """The exact function the live analysis path calls (adaptive routing)."""

    columns = list(feature_columns) if feature_columns else None

    def _forecast(sku: str, history: pd.Series, horizon: int) -> Forecast | None:
        values, method = adaptive_forecast(
            sku=sku,
            demand_series=history,
            horizon=horizon,
            model=model,
            feature_columns=columns,
            routing_service=routing_service,
            include_routing=False,
        )
        return Forecast(tuple(float(v) for v in values[:horizon]), method)

    return _forecast


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def prepare_sku_series(
    daily: pd.DataFrame,
    *,
    skus: Iterable[str] | None = None,
    pad_to: pd.Timestamp | None = None,
) -> dict[str, pd.Series]:
    """Continuous daily series per SKU, zero-filled between (and, if asked, after) sales.

    ``pad_to`` extends every series with zeros to a common end date. Without it
    a SKU that stopped selling looks like it "ended" earlier and its trailing
    zero days silently disappear from both classification and scoring.
    """
    frame = daily[["StockCode", "date", "demand"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if skus is not None:
        frame = frame[frame["StockCode"].isin(set(skus))]
    end = pd.Timestamp(pad_to) if pad_to is not None else None

    out: dict[str, pd.Series] = {}
    for sku, group in frame.groupby("StockCode", sort=False):
        group = group.sort_values("date")
        last = max(group["date"].max(), end) if end is not None else group["date"].max()
        index = pd.date_range(group["date"].min(), last, freq="D")
        series = group.set_index("date")["demand"].astype(float).reindex(index, fill_value=0.0)
        out[str(sku)] = series
    return out


def select_eval_skus(
    daily: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    min_active_days: int = 60,
    max_skus: int | None = 500,
) -> list[str]:
    """Rank SKUs using only data strictly before ``cutoff`` (no peeking at the test period)."""
    frame = daily[pd.to_datetime(daily["date"]) <= pd.Timestamp(cutoff)]
    stats = frame.groupby("StockCode").agg(active_days=("date", "nunique"), total=("demand", "sum"))
    eligible = stats[stats["active_days"] >= min_active_days].sort_values(
        ["total"], ascending=False, kind="mergesort"
    )
    if max_skus is not None:
        eligible = eligible.head(max_skus)
    return [str(s) for s in eligible.index.tolist()]


def origin_positions(series: pd.Series, config: BacktestConfig, *, dataset_end: pd.Timestamp) -> list[int]:
    """Positions ``t`` such that history = ``series.iloc[:t]`` and targets = ``iloc[t:t+H]``.

    The first target day must fall after ``dataset_end - eval_days`` and the last
    target day must be recorded, so every scored day is real demand.
    """
    cutoff = pd.Timestamp(dataset_end) - pd.Timedelta(days=config.eval_days)
    first_target = cutoff + pd.Timedelta(days=1)
    n = len(series)
    positions = np.nonzero(series.index >= first_target)[0]
    if positions.size == 0:
        return []
    t0 = int(positions[0])
    if t0 < config.min_history_days:
        t0 = config.min_history_days
    return list(range(t0, n - config.horizon + 1, config.stride))


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def forecast_at_origin(
    series: pd.Series,
    t: int,
    horizon: int,
    forecasters: Mapping[str, Forecaster],
    *,
    sku: str = "sku",
    history_window: int = PRODUCTION_HISTORY_DAYS,
) -> dict[str, Forecast | None]:
    """Forecast from origin ``t`` using only ``series.iloc[:t]``.

    Anything at or after position ``t`` is invisible to the forecasters; the
    leakage tests mutate and truncate it to prove that.
    """
    history = series.iloc[:t].tail(history_window)
    return {name: fn(sku, history, horizon) for name, fn in forecasters.items()}


@dataclass
class _Sums:
    n_days: int = 0
    n_origins: int = 0
    sum_actual: float = 0.0
    sum_abs_err: float = 0.0
    sum_sq_err: float = 0.0
    sum_err: float = 0.0
    lt_sum_actual: float = 0.0
    lt_sum_abs_err: float = 0.0
    mase_num: float = 0.0
    mase_den: int = 0
    skus: set = field(default_factory=set)

    def add(self, sku: str, actual: np.ndarray, pred: np.ndarray, scale: float | None) -> None:
        err = pred - actual
        self.n_days += int(actual.size)
        self.n_origins += 1
        self.sum_actual += float(actual.sum())
        self.sum_abs_err += float(np.abs(err).sum())
        self.sum_sq_err += float((err ** 2).sum())
        self.sum_err += float(err.sum())
        self.lt_sum_actual += float(actual.sum())
        self.lt_sum_abs_err += abs(float(pred.sum()) - float(actual.sum()))
        if scale and scale > 0:
            self.mase_num += float(np.abs(err).mean()) / scale * actual.size
            self.mase_den += int(actual.size)
        self.skus.add(sku)

    def as_metrics(self) -> dict[str, Any]:
        def ratio(num: float, den: float) -> float | None:
            return round(num / den, 4) if den > 0 else None

        return {
            "wape": ratio(self.sum_abs_err, self.sum_actual),
            "wape_lead_time_sum": ratio(self.lt_sum_abs_err, self.lt_sum_actual),
            "mae": round(self.sum_abs_err / self.n_days, 4) if self.n_days else None,
            "rmse": round(math.sqrt(self.sum_sq_err / self.n_days), 4) if self.n_days else None,
            "bias": round(self.sum_err / self.n_days, 4) if self.n_days else None,
            "bias_ratio": ratio(self.sum_err, self.sum_actual),
            "mase": round(self.mase_num / self.mase_den, 4) if self.mase_den else None,
            "n_skus": len(self.skus),
            "n_origins": self.n_origins,
            "n_test_points": self.n_days,
        }


def _naive_scale(history: pd.Series) -> float | None:
    values = history.to_numpy(dtype=float)
    if values.size < 2:
        return None
    scale = float(np.mean(np.abs(np.diff(values))))
    return scale if scale > 0 else None


@dataclass
class BacktestResult:
    config: BacktestConfig
    aggregates: dict[str, dict[str, dict[str, Any]]]
    per_sku: dict[str, dict[str, dict[str, dict[str, float]]]]
    n_skus: int
    n_origins: int
    n_skipped_origins: int
    routed_method_counts: dict[str, int]
    period: dict[str, str]
    methods: list[str]

    def to_payload(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "evaluation_mode": EVALUATION_MODE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "horizon_days": self.config.horizon,
            "eval_days": self.config.eval_days,
            "stride": self.config.stride,
            "history_window_days": self.config.history_window,
            "n_skus_evaluated": self.n_skus,
            "n_origins": self.n_origins,
            "n_skipped_origins": self.n_skipped_origins,
            "period": self.period,
            "models": self.methods,
            "routed_method_counts": self.routed_method_counts,
            "aggregates": self.aggregates,
        }
        if extra:
            payload.update(extra)
        return payload

    def strongest_baseline(self, demand_class: str, metric: str = "wape_lead_time_sum") -> tuple[str, float] | None:
        return strongest_baseline(self.aggregates.get(demand_class) or {}, metric)


def strongest_baseline(
    class_metrics: Mapping[str, Mapping[str, Any]],
    metric: str = "wape_lead_time_sum",
    baselines: Sequence[str] = REFERENCE_METHODS,
) -> tuple[str, float] | None:
    """Lowest-error reference method for one demand class (predict-zero counts)."""
    scored = [
        (name, float(class_metrics[name][metric]))
        for name in baselines
        if name in class_metrics and class_metrics[name].get(metric) is not None
    ]
    if not scored:
        return None
    return min(scored, key=lambda item: (item[1], item[0]))


def run_backtest(
    series_by_sku: Mapping[str, pd.Series],
    forecasters: Mapping[str, Forecaster],
    config: BacktestConfig,
    *,
    dataset_end: pd.Timestamp | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """Run every forecaster over every rolling origin of every SKU."""

    if not series_by_sku:
        raise ValueError("No series supplied to run_backtest.")
    merged: dict[str, Forecaster] = {**reference_forecasters(), **dict(forecasters)}
    methods = list(merged.keys())
    dataset_end = pd.Timestamp(dataset_end) if dataset_end is not None else max(s.index[-1] for s in series_by_sku.values())

    sums: dict[str, dict[str, _Sums]] = defaultdict(lambda: defaultdict(_Sums))
    per_sku_sums: dict[str, dict[str, dict[str, _Sums]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(_Sums)))
    routed_counts: dict[str, int] = defaultdict(int)
    skipped = 0
    n_origins = 0
    first_origin_date: pd.Timestamp | None = None
    last_origin_date: pd.Timestamp | None = None
    used_skus: set[str] = set()

    total = len(series_by_sku)
    for position, (sku, series) in enumerate(series_by_sku.items(), start=1):
        for t in origin_positions(series, config, dataset_end=dataset_end):
            actual = series.iloc[t:t + config.horizon].to_numpy(dtype=float)
            if actual.size != config.horizon:
                continue
            outputs = forecast_at_origin(
                series, t, config.horizon, merged, sku=sku, history_window=config.history_window
            )
            if any(o is None or len(o.values) < config.horizon for o in outputs.values()):
                skipped += 1
                continue

            history = series.iloc[:t].tail(config.history_window)
            demand_class = classify_sku_demand_pattern(history)
            scale = _naive_scale(history)
            for name, out in outputs.items():
                pred = np.asarray(out.values[: config.horizon], dtype=float)
                sums["all"][name].add(sku, actual, pred, scale)
                sums[demand_class][name].add(sku, actual, pred, scale)
                per_sku_sums[sku][demand_class][name].add(sku, actual, pred, scale)
                if name == PRODUCTION_ROUTED and out.method:
                    routed_counts[out.method] += 1

            n_origins += 1
            used_skus.add(sku)
            origin_date = series.index[t - 1]
            first_origin_date = origin_date if first_origin_date is None else min(first_origin_date, origin_date)
            last_origin_date = origin_date if last_origin_date is None else max(last_origin_date, origin_date)
        if progress:
            progress(position, total)

    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    for bucket in ("all", *DEMAND_CLASSES):
        if bucket not in sums:
            continue
        aggregates[bucket] = {name: sums[bucket][name].as_metrics() for name in methods if name in sums[bucket]}

    per_sku_out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for sku, by_class in per_sku_sums.items():
        per_sku_out[sku] = {
            cls: {
                name: {
                    "abs_err": s.sum_abs_err,
                    "actual": s.sum_actual,
                    "lt_abs_err": s.lt_sum_abs_err,
                    "lt_actual": s.lt_sum_actual,
                }
                for name, s in by_method.items()
            }
            for cls, by_method in by_class.items()
        }

    period = {
        "dataset_end": str(dataset_end.date()),
        "train_cutoff": str((dataset_end - pd.Timedelta(days=config.eval_days)).date()),
        "first_origin": str(first_origin_date.date()) if first_origin_date is not None else "",
        "last_origin": str(last_origin_date.date()) if last_origin_date is not None else "",
    }
    return BacktestResult(
        config=config,
        aggregates=aggregates,
        per_sku=per_sku_out,
        n_skus=len(used_skus),
        n_origins=n_origins,
        n_skipped_origins=skipped,
        routed_method_counts=dict(routed_counts),
        period=period,
        methods=methods,
    )


# --------------------------------------------------------------------------
# Uncertainty: SKU-cluster bootstrap
# --------------------------------------------------------------------------

def bootstrap_wape_ci(
    per_sku: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    *,
    demand_class: str,
    method: str,
    metric: str = "wape_lead_time_sum",
    reference: str | None = None,
    n_boot: int = 500,
    seed: int = 7,
) -> dict[str, float] | None:
    """95% CI for a method's WAPE, or for its (paired) difference vs ``reference``.

    Origins within a SKU are strongly autocorrelated, so the SKU, not the origin,
    is the independent unit that gets resampled.
    """
    err_key, act_key = ("lt_abs_err", "lt_actual") if metric == "wape_lead_time_sum" else ("abs_err", "actual")

    def totals(name: str) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}
        for sku, by_class in per_sku.items():
            if demand_class == "all":
                cells = [c[name] for c in by_class.values() if name in c]
            else:
                cell = (by_class.get(demand_class) or {}).get(name)
                cells = [cell] if cell else []
            errs = sum(c[err_key] for c in cells)
            acts = sum(c[act_key] for c in cells)
            if acts > 0:
                result[sku] = (errs, acts)
        return result

    a = totals(method)
    b = totals(reference) if reference else None
    skus = sorted(a if b is None else set(a) & set(b))
    if len(skus) < 5:
        return None

    ea = np.array([a[s][0] for s in skus]); aa = np.array([a[s][1] for s in skus])
    eb = ab = None
    if b is not None:
        eb = np.array([b[s][0] for s in skus]); ab = np.array([b[s][1] for s in skus])

    def statistic(idx: np.ndarray) -> float:
        value = ea[idx].sum() / aa[idx].sum()
        if eb is not None:
            value -= eb[idx].sum() / ab[idx].sum()
        return float(value)

    rng = np.random.default_rng(seed)
    draws = [statistic(rng.integers(0, len(skus), len(skus))) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "point": statistic(np.arange(len(skus))),
        "lo": float(lo),
        "hi": float(hi),
        "n_skus": len(skus),
    }
