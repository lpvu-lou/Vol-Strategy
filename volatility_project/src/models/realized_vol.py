from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR

# Realized volatility estimation using rolling windows of returns
def realized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

# Compute rolling realized volatility estimates from a series of returns
def rolling_realized_volatility(
    returns: pd.Series,
    window: int,
    volatility_type: Literal["std"] = "std",
    volatility_kwargs: Optional[dict] = None,
) -> pd.Series:
    if volatility_type != "std":
        raise ValueError("Unsupported volatility_type")
    return returns.rolling(window).apply(realized_volatility, raw=False, **(volatility_kwargs or {}))
