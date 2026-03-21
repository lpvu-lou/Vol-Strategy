from __future__ import annotations

from typing import Literal

import pandas as pd

from src.utils.helpers import check_is_true


def select_options(
    df_options: pd.DataFrame,
    call_or_put: Literal["C", "P"],
    strike_col: Literal["delta", "strike", "moneyness"],
    strike_target: float,
    day_to_expiry_target: int,
) -> pd.DataFrame:
    selected_maturity = select_closest_maturity(df_options, day_to_expiry_target=day_to_expiry_target)
    return select_closest_strike(
        selected_maturity,
        strike_col=strike_col,
        target=strike_target,
        call_put=call_or_put,
    ).drop_duplicates(["date", "ticker"])


def select_closest_maturity(df_option: pd.DataFrame, day_to_expiry_target: int) -> pd.DataFrame:
    return _select_close_target(
        df_option,
        target_col="day_to_expiration",
        target=day_to_expiry_target,
        groupby_cols=["date", "ticker"],
    )


def select_closest_strike(
    df_option: pd.DataFrame,
    strike_col: Literal["delta", "strike", "moneyness"],
    target: float,
    call_put: Literal["C", "P"] = "C",
) -> pd.DataFrame:
    check_is_true(call_put in ["C", "P"], "call_put must be either P or C")
    filtered = df_option[df_option["call_put"] == call_put].copy()
    return _select_close_target(
        filtered,
        target_col=strike_col,
        target=target,
        groupby_cols=["date", "ticker", "expiration"],
    )


def _select_close_target(df: pd.DataFrame, target_col: str, target: float | int, groupby_cols: list[str]) -> pd.DataFrame:
    missing_cols = set(groupby_cols + [target_col]).difference(df.columns)
    check_is_true(len(missing_cols) == 0, f"Missing columns: {missing_cols}")
    df = df.copy()
    df["criterion"] = (df[target_col] - target).abs()
    df_min = df.groupby(groupby_cols)[["criterion"]].min().reset_index()
    return df.merge(df_min, on=[*groupby_cols, "criterion"], how="inner").drop(columns=["criterion"])
