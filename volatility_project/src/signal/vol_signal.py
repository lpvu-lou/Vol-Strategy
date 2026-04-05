from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.realized_vol import rolling_realized_volatility
from src.utils.helpers import ensure_datetime_indexed_frame, merge_on_date

# Compute a rolling realised volatility series from spot prices
def rolling_realized_vol_benchmark(spot: pd.Series, window: int = 21) -> pd.DataFrame:
    spot_series = pd.Series(spot).dropna().astype(float)
    spot_series.index = pd.to_datetime(spot_series.index)
    log_returns = np.log(spot_series).diff()
    rolling_vol = rolling_realized_volatility(log_returns, window=window, volatility_type="std")
    return pd.DataFrame(
        {
            "date": rolling_vol.index,
            "spot": spot_series.reindex(rolling_vol.index).to_numpy(),
            "log_return": log_returns.reindex(rolling_vol.index).to_numpy(),
            "rolling_realized_vol": rolling_vol.to_numpy(),
        }
    ).dropna(subset=["rolling_realized_vol"])

# Compute a weight-averaged implied volatility reference for the option strategy
def compute_strategy_iv_reference(
    df_positions: pd.DataFrame,
    df_options: pd.DataFrame,
    *,
    date_col: str = "date",
    iv_col: str = "implied_volatility",
    exclude_leg_names: tuple[str, ...] = ("DELTA_HEDGING",),
) -> pd.DataFrame:
    positions = ensure_datetime_indexed_frame(df_positions, date_col=date_col)
    options = ensure_datetime_indexed_frame(df_options, date_col=date_col)
    if exclude_leg_names:
        positions = positions.loc[~positions["leg_name"].isin(exclude_leg_names)].copy()
    merged = positions.merge(options[[date_col, "option_id", iv_col]], on=[date_col, "option_id"], how="left")

    # Use absolute weights so short legs contribute positively to the average
    merged["abs_weight"] = merged["weight"].abs()
    merged["weighted_iv"] = merged["abs_weight"] * merged[iv_col]
    return (
        merged.groupby(date_col, as_index=False)[["weighted_iv", "abs_weight"]]
        .sum()
        .assign(iv_reference=lambda df: np.where(df["abs_weight"] > 0, df["weighted_iv"] / df["abs_weight"], np.nan))[[
            date_col,
            "iv_reference",
        ]]
    )

# Construct the volatility trading signal by comparing IV to a model forecast 
def build_vol_signal(
    df_iv: pd.DataFrame,
    df_sigma_hat: pd.DataFrame,
    *,
    signal_definition: str = "iv_minus_sigma",
    winsorize_quantiles: tuple[float, float] | None = None,
) -> pd.DataFrame:
    sigma_col = "sigma_hat"
    if "forecast_sigma_hat" in df_sigma_hat.columns:
        sigma_col = "forecast_sigma_hat"
    elif "horizon_sigma_hat" in df_sigma_hat.columns:
        sigma_col = "horizon_sigma_hat"
    signal = merge_on_date(df_iv[["date", "iv_reference"]], df_sigma_hat[["date", sigma_col]], how="inner")
    signal = signal.rename(columns={sigma_col: "sigma_hat"})
    if signal_definition == "iv_minus_sigma":
        signal["vol_signal"] = signal["iv_reference"] - signal["sigma_hat"]
    elif signal_definition == "sigma_minus_iv":
        signal["vol_signal"] = signal["sigma_hat"] - signal["iv_reference"]
    else:
        raise ValueError("signal_definition must be either 'sigma_minus_iv' or 'iv_minus_sigma'")
    signal["signal_definition"] = signal_definition
    if winsorize_quantiles is not None:
        lower, upper = winsorize_quantiles
        signal["vol_signal"] = signal["vol_signal"].clip(signal["vol_signal"].quantile(lower), signal["vol_signal"].quantile(upper))
    return signal

# Shift the signal forward by lag_business_days to simulate realistic trade execution
def lag_signal_for_trading(signal_df: pd.DataFrame, lag_business_days: int = 1) -> pd.DataFrame:
    """Observe at t and trade at t+lag."""

    out = ensure_datetime_indexed_frame(signal_df)
    out["entry_date"] = out["date"] + pd.offsets.BusinessDay(lag_business_days)
    return out


def map_signal_to_trade_entries(
    df_positions: pd.DataFrame,
    signal_df: pd.DataFrame,
    *,
    allocation_col: str = "allocation_multiplier",
    lag_business_days: int = 1,
) -> pd.DataFrame:
    """Attach one allocation value per trade entry date, then keep it fixed through the life of the trade."""

    signal_lagged = lag_signal_for_trading(signal_df, lag_business_days=lag_business_days)
    entry_alloc = signal_lagged[["entry_date", allocation_col]].drop_duplicates(subset=["entry_date"]).copy()
    positions = df_positions.merge(entry_alloc, on="entry_date", how="left")
    positions[allocation_col] = positions[allocation_col].fillna(1.0)
    return positions
