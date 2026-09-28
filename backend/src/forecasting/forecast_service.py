"""Recursive multi-step demand forecasting.

``recursive_forecast`` predicts one day, appends that prediction to the
series, and rebuilds *every* feature for the next day from the extended
series via ``features.inference_features.feature_vector``. Lags, rolling
statistics, and calendar features therefore all stay correct and advance with
the date across the whole horizon — each step sees exactly the feature values
the model would see if that prediction were a real observation.

Only ``history`` is ever read; nothing after it is used, so a forecast at a
given origin is identical regardless of what data exists (or doesn't) beyond
that point. See ``tests/test_recursive_forecast.py`` for the leakage and
feature-parity proofs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from features.inference_features import feature_vector


def recursive_forecast(
    model: Any,
    history: pd.Series,
    feature_columns: Sequence[str],
    horizon: int = 7,
) -> list[float] | None:
    """Forecast ``horizon`` days after ``history`` ends, or ``None`` if features can't be built."""
    columns = list(feature_columns)
    values = np.nan_to_num(pd.to_numeric(history, errors="coerce").to_numpy(dtype=float), nan=0.0).tolist()
    if isinstance(history.index, pd.DatetimeIndex) and len(history.index) and not history.index.hasnans:
        last_date = pd.DatetimeIndex(history.index)[-1]
    else:
        last_date = pd.Timestamp.utcnow().normalize()

    forecasts: list[float] = []
    for step in range(1, horizon + 1):
        row = feature_vector(values, last_date + pd.Timedelta(days=step), columns)
        if row is None:
            return None
        prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=columns))[0]))
        forecasts.append(prediction)
        values.append(prediction)
    return forecasts
