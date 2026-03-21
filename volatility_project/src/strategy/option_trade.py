from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.data.option_loader import OptionLoader
from src.data.rates_loader import USRatesLoader
from src.strategy.option_selection import select_options
from src.utils.helpers import check_is_true, ffill_options_data
from src.config import DAYS_PER_YEAR, TENOR_TO_PERIOD


def interpolate_rates(eval_tenor: float, tenors: pd.Series | np.ndarray, rate_curve: pd.Series | np.ndarray) -> float:
    tenors = np.asarray(tenors)
    rate_curve = np.asarray(rate_curve)
    check_is_true(len(tenors) == len(rate_curve), "Tenors and rate curve must have the same length.")
    if eval_tenor <= tenors.min():
        return rate_curve[tenors.argmin()]
    if eval_tenor >= tenors.max():
        return rate_curve[tenors.argmax()]
    idx_above = tenors[tenors >= eval_tenor].argmin()
    idx_below = tenors[tenors <= eval_tenor].argmax()
    tenor_above, tenor_below = tenors[idx_above], tenors[idx_below]
    rate_above, rate_below = rate_curve[idx_above], rate_curve[idx_below]
    weight_above = (eval_tenor - tenor_below) / (tenor_above - tenor_below)
    return (1 - weight_above) * rate_below + weight_above * rate_above


def compute_forward(df_options: pd.DataFrame, df_rates: pd.DataFrame) -> pd.DataFrame:
    def _compute_values(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        dte = group["day_to_expiration"].iloc[0] / DAYS_PER_YEAR
        tenors = group[list(TENOR_TO_PERIOD.keys())].columns.map(TENOR_TO_PERIOD).to_numpy()
        rate_curve = group[list(TENOR_TO_PERIOD.keys())].drop_duplicates().to_numpy().reshape(-1)
        group["risk_free_rate"] = interpolate_rates(dte, tenors=tenors, rate_curve=rate_curve)
        return group

    df = df_options.merge(df_rates, on="date", how="left")
    df = df.groupby(["date", "expiration"], group_keys=False).apply(_compute_values).reset_index(drop=True)
    df["forward"] = df["spot"] * np.exp(df["risk_free_rate"] * df["day_to_expiration"] / DAYS_PER_YEAR)
    df_forward = df.groupby(["ticker", "date", "expiration"])[["forward"]].first().ffill().reset_index()
    return df.drop(columns=list(TENOR_TO_PERIOD.keys()) + ["forward"]).merge(
        df_forward,
        how="left",
        on=["ticker", "date", "expiration"],
    )


class OptionTradeABC(ABC):
    _REQUIRED_COLUMNS = ["date", "option_id", "expiration", "delta", "strike", "moneyness", "call_put", "spot", "ticker"]

    @classmethod
    def generate_trades(
        cls,
        start_date: datetime,
        end_date: datetime,
        tickers: list[str] | str,
        legs: list[dict],
        cost_neutral: bool = False,
        hedging_args: Optional[dict] = None,
    ) -> pd.DataFrame:
        df_trades_daily = cls._generate_trades(start_date, end_date, tickers=tickers, legs=legs, cost_neutral=cost_neutral)
        return cls._hedge_trades(df_trades_daily, **(hedging_args or {}))[["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]]

    @classmethod
    def _generate_trades(cls, start_date: datetime, end_date: datetime, tickers: list[str] | str, legs: list[dict], cost_neutral: bool = False) -> pd.DataFrame:
        df_options = cls._load_option_data(start_date, end_date, process_kwargs={"ticker": tickers})
        df_trades = cls._select_options(df_options, legs, cost_neutral=cost_neutral)
        df_trades_daily = cls._convert_trades_to_timeseries(df_trades)
        df_trades_daily = df_trades_daily.merge(df_options, on=["date", "option_id", "ticker"], how="left")
        df_trades_daily = df_trades_daily[df_trades_daily["date"].between(start_date, end_date)]
        df_trades_daily = df_trades_daily.drop_duplicates(subset=["date", "leg_name", "option_id"])
        df_trades_daily = ffill_options_data(df_trades_daily)
        if "risk_free_rate" not in df_trades_daily.columns:
            df_rates = USRatesLoader.load_data(df_trades_daily["date"].min(), df_trades_daily["date"].max())
            df_trades_daily = compute_forward(df_trades_daily, df_rates)
        return df_trades_daily

    @classmethod
    def _load_option_data(cls, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        option_df = cls.load_data(start_date, end_date, **kwargs)
        missing_cols = set(cls._REQUIRED_COLUMNS).difference(option_df.columns)
        check_is_true(len(missing_cols) == 0, f"Option data is missing required columns: {missing_cols}")
        return option_df

    @classmethod
    @abstractmethod
    def load_data(cls, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _select_options(cls, df_options: pd.DataFrame, legs: list[dict], cost_neutral: bool = False) -> pd.DataFrame:
        df_list = []
        for leg in deepcopy(legs):
            leg_name = leg.pop("leg_name", "")
            weight = leg.pop("weight", np.nan)
            rebal_week_day = leg.pop("rebal_week_day", [1])
            selected = select_options(df_options, **leg)
            selected["leg_name"] = leg_name
            selected["weight"] = (weight / selected["spot"].where(selected["spot"] != 0, np.nan)).ffill()
            selected = selected[selected["date"].dt.day_of_week.isin(rebal_week_day)]
            df_list.append(selected.rename(columns={"date": "entry_date"}))
        df = pd.concat(df_list)
        if cost_neutral:
            df = cls._neutralize_cost(df)
        return df[["entry_date", "option_id", "expiration", "leg_name", "weight", "ticker"]].drop_duplicates(subset=["entry_date", "leg_name", "ticker"])

    @classmethod
    def _neutralize_cost(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        df_trades_cp = df_trades.copy()
        df_trades_cp["premium"] = df_trades_cp["weight"] * df_trades_cp["mid"]
        df_trades_cp["L/S"] = np.where(df_trades_cp["weight"] > 0, "Long", "Short")
        trade_pivot = df_trades_cp.pivot_table(index=["entry_date", "ticker"], columns="L/S", values="premium", aggfunc="sum")
        trade_pivot["missing_premium"] = -trade_pivot["Long"] - trade_pivot["Short"]
        trade_pivot["scaling_factor"] = np.where(
            trade_pivot["missing_premium"] < 0,
            (trade_pivot["Short"] + trade_pivot["missing_premium"]) / trade_pivot["Short"],
            (trade_pivot["Long"] + trade_pivot["missing_premium"]) / trade_pivot["Long"],
        )
        df_trades_cp = df_trades_cp.merge(
            trade_pivot.reset_index()[["entry_date", "ticker", "scaling_factor", "missing_premium"]],
            on=["entry_date", "ticker"],
            how="left",
        )
        df_trades_cp["weight"] = np.where(
            ((df_trades_cp["missing_premium"] < 0) & (df_trades_cp["weight"] < 0))
            | ((df_trades_cp["missing_premium"] > 0) & (df_trades_cp["weight"] > 0)),
            df_trades_cp["weight"] * df_trades_cp["scaling_factor"],
            df_trades_cp["weight"],
        )
        return df_trades_cp

    @classmethod
    def _convert_trades_to_timeseries(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        df_trades_cp = df_trades.copy()
        df_trades_cp["date"] = df_trades_cp.apply(lambda r: pd.date_range(start=r["entry_date"], end=r["expiration"], freq="B"), axis=1)
        df_trades_cp = df_trades_cp.explode("date").reset_index(drop=True)
        return df_trades_cp[["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]]

    @classmethod
    def _hedge_trades(cls, df_trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df_trades


class OptionTrade(OptionLoader, OptionTradeABC):
    pass


class DeltaHedgedOptionTrade(OptionTrade):
    @classmethod
    def _hedge_trades(cls, df_trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_hedge = (
            df_trades.groupby(["date", "ticker", "entry_date"], group_keys=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "option_id": x["ticker"].iloc[0],
                        "expiration": x["date"].iloc[0] + pd.offsets.BusinessDay(n=1),
                        "leg_name": "DELTA_HEDGING",
                        "weight": -(x["delta"] * x["weight"]).sum(),
                    }
                )
            )
            .reset_index()
        )
        return pd.concat([df_trades, df_hedge], ignore_index=True).sort_values(by=["date", "option_id"]).reset_index(drop=True)
