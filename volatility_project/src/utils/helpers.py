from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def check_is_true(condition: bool, message: Optional[str] = None) -> None:
    if not condition:
        raise ValueError(message or "Condition is not true.")


def ffill_options_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = {"option_id", "date"}.difference(df.columns)
    check_is_true(len(missing_cols) == 0, f"Missing columns: {missing_cols}")
    sorted_df = df.sort_values(by=["option_id", "date"]).copy()
    filled = sorted_df.groupby("option_id", group_keys=False).ffill()
    # Preserve the original identifier column after the grouped forward-fill.
    filled["option_id"] = sorted_df["option_id"].to_numpy()
    return filled


def ensure_datetime_indexed_frame(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values(date_col).reset_index(drop=True)


def merge_on_date(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    return ensure_datetime_indexed_frame(left).merge(
        ensure_datetime_indexed_frame(right),
        on="date",
        how=how,
    )


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))


def summarize_series(series: pd.Series, percentiles: Sequence[float] = (0.05, 0.5, 0.95)) -> pd.Series:
    summary = {
        "count": float(series.count()),
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    }
    for percentile in percentiles:
        summary[f"q_{int(percentile * 100):02d}"] = series.quantile(percentile)
    return pd.Series(summary)


def annualized_volatility_from_daily(daily_vol: pd.Series | float | np.ndarray):
    return daily_vol * np.sqrt(252)
