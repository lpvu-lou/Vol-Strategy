from __future__ import annotations

import numpy as np
import pandas as pd


def linear_allocation(signal: pd.Series, slope: float = 5.0, floor: float = 0.0, cap: float = 2.0, base: float = 1.0) -> pd.Series:
    allocation = base + slope * signal.fillna(0.0)
    return allocation.clip(lower=floor, upper=cap)


def tanh_allocation(
    signal: pd.Series,
    scale: float = 10.0,
    leverage_cap: float = 2.0,
    base: float = 1.0,
    floor: float = 0.0,
    *,
    increasing: bool = True,
) -> pd.Series:
    centered = np.tanh(scale * signal.fillna(0.0))
    if not increasing:
        centered = -centered
    upper_span = max(leverage_cap - base, 0.0)
    lower_span = max(base - floor, 0.0)
    allocation = np.where(centered >= 0, base + upper_span * centered, base + lower_span * centered)
    return pd.Series(allocation, index=signal.index, name="allocation_multiplier")


def bucket_allocation(
    signal: pd.Series,
    low: float = -0.02,
    high: float = 0.02,
    low_allocation: float = 0.5,
    mid_allocation: float = 1.0,
    high_allocation: float = 1.5,
    *,
    increasing: bool = True,
) -> pd.Series:
    signal_filled = signal.fillna(0.0)
    if increasing:
        allocation = np.where(signal_filled <= low, low_allocation, np.where(signal_filled >= high, high_allocation, mid_allocation))
    else:
        allocation = np.where(signal_filled <= low, high_allocation, np.where(signal_filled >= high, low_allocation, mid_allocation))
    return pd.Series(allocation, index=signal.index, name="allocation_multiplier")
