from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR
from src.data_loader.option_loader import OptionLoader
from src.data_loader.rates_loader import USRatesLoader
from src.models.black_scholes_greeks import compute_black_scholes_greeks
from src.strategy.option_trade import compute_forward
from src.utils.helpers import check_is_true, ffill_options_data


class StrategyBacktester:
    _BACKTEST_COLS = ["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]
    _PNL_COLS = [
        "pnl",
        "model_pnl",
        "tcost_pnl",
        "delta_pnl",
        "gamma_pnl",
        "theta_pnl",
        "vega_pnl",
        "residual_pnl",
        "leverage",
        "cashflow",
        "cash_interest",
    ]

    def __init__(self, df_positions: pd.DataFrame) -> None:
        missing_cols = set(self._BACKTEST_COLS).difference(df_positions.columns)
        check_is_true(len(missing_cols) == 0, f"Positions data is missing required columns: {missing_cols}")
        check_is_true(len(df_positions) > 2, "Positions data is empty or too small to run backtest.")
        self._df_positions = df_positions[self._BACKTEST_COLS].copy()
        self._is_backtested = False
        self._df_pnl = pd.DataFrame()
        self._df_nav = pd.DataFrame()
        self._df_metainfo = pd.DataFrame()
        self._df_drifted_positions = pd.DataFrame()

    def compute_backtest(self, tcost_args: Optional[dict[str, Any]] = None) -> "StrategyBacktester":
        tcost_args = dict(tcost_args or {})
        spot_spread_bps = float(tcost_args.pop("spot_spread_bps", 1.0))
        recompute_greeks = bool(tcost_args.pop("recompute_greeks", True))
        df_positions_raw = self._preprocess_positions(
            self._df_positions[self._BACKTEST_COLS],
            spot_spread_bps=spot_spread_bps,
            recompute_greeks=recompute_greeks,
        )
        df_positions_raw["mid_model"] = df_positions_raw["mid"]
        df_positions = self.apply_tcost(df_positions_raw, **tcost_args).sort_values(["option_id", "date"])

        df_positions["dv"] = df_positions.groupby(["option_id"])["mid"].diff().fillna(0.0)
        df_positions["dv_model"] = df_positions.groupby(["option_id"])["mid_model"].diff().fillna(0.0)
        df_positions["dsigma"] = df_positions.groupby(["option_id"])["implied_volatility"].diff().fillna(0.0)
        df_positions["dS"] = df_positions.groupby(["option_id"])["spot"].diff().fillna(0.0)
        # The option dataset's theta is already expressed as a one-day carry.
        df_positions["dt"] = 1.0
        df_positions["prev_theta"] = df_positions.groupby("option_id")["theta"].shift(1).bfill()
        df_positions["prev_gamma"] = df_positions.groupby("option_id")["gamma"].shift(1).bfill()
        df_positions["prev_delta"] = df_positions.groupby("option_id")["delta"].shift(1).bfill()
        df_positions["prev_vega"] = df_positions.groupby("option_id")["vega"].shift(1).bfill()
        df_positions["obs_date"] = df_positions["entry_date"] - pd.Timedelta(days=1)

        start_idx = df_positions["date"].min() - pd.Timedelta(days=1)
        df_pnl = pd.DataFrame([[0.0] * len(self._PNL_COLS)], columns=self._PNL_COLS, index=[start_idx])
        df_nav = pd.DataFrame([[1.0, 1.0, 0.0]], columns=["NAV", "cash_account", "position_value"], index=[start_idx])
        drifted_positions = []

        for current_date in sorted(df_positions["date"].unique()):
            df_day = df_positions[df_positions["date"] == current_date].copy()
            df_day = df_day.merge(df_nav[["NAV"]], left_on="obs_date", right_index=True, how="left")
            df_day["scaled_weight"] = (df_day["weight"] * df_day["NAV"]).fillna(df_day["weight"])
            df_day["effective_weight"] = df_day["scaled_weight"]
            df_day["pnl"] = df_day["effective_weight"] * df_day["dv"]
            df_day["model_pnl"] = df_day["effective_weight"] * df_day["dv_model"]
            df_day["tcost_pnl"] = df_day["pnl"] - df_day["model_pnl"]
            df_day["gamma_pnl"] = 0.5 * df_day["effective_weight"] * df_day["dS"] ** 2 * df_day["prev_gamma"]
            df_day["delta_pnl"] = df_day["effective_weight"] * df_day["dS"] * df_day["prev_delta"]
            df_day["theta_pnl"] = df_day["effective_weight"] * df_day["dt"] * df_day["prev_theta"]
            df_day["vega_pnl"] = df_day["effective_weight"] * df_day["dsigma"] * df_day["prev_vega"]
            df_day["residual_pnl"] = (
                df_day["model_pnl"]
                - df_day["delta_pnl"]
                - df_day["gamma_pnl"]
                - df_day["theta_pnl"]
                - df_day["vega_pnl"]
            )
            df_day["leverage"] = df_day["effective_weight"] * df_day["spot"]
            df_day["cashflow"] = 0.0
            df_day.loc[df_day["entry_date"] == df_day["date"], "cashflow"] = -df_day["effective_weight"] * df_day["mid"]
            df_day.loc[df_day["expiration"] == df_day["date"], "cashflow"] = df_day["effective_weight"] * df_day["mid"]
            previous_state = df_nav.iloc[-1]
            daily_rate = float(df_day["risk_free_rate"].dropna().mean()) if "risk_free_rate" in df_day.columns and not df_day["risk_free_rate"].dropna().empty else 0.0
            cash_interest = float(previous_state["cash_account"]) * (np.exp(daily_rate / TRADING_DAYS_PER_YEAR) - 1.0)
            df_day["cash_interest"] = 0.0
            df_day.loc[df_day.index[0], "cash_interest"] = cash_interest
            df_pnl = pd.concat([df_pnl, df_day.groupby("date")[self._PNL_COLS].sum()])
            total_pnl = float(df_pnl.loc[current_date, "pnl"])
            total_cashflow = float(df_pnl.loc[current_date, "cashflow"])
            total_cash_interest = float(df_pnl.loc[current_date, "cash_interest"])
            current_nav = float(previous_state["NAV"]) + total_pnl + total_cash_interest
            current_cash = float(previous_state["cash_account"]) + total_cash_interest + total_cashflow
            current_position_value = current_nav - current_cash
            df_nav.loc[current_date] = [current_nav, current_cash, current_position_value]
            drifted_positions.append(df_day)

        self._is_backtested = True
        self._df_pnl = df_pnl.drop(columns=["leverage", "cashflow"]).copy()
        self._df_nav = df_nav.copy()
        self._df_metainfo = df_pnl[["leverage", "cashflow", "cash_interest"]].join(df_nav[["cash_account", "position_value"]], how="left")
        self._df_drifted_positions = pd.concat(drifted_positions).reset_index(drop=True)
        return self

    @classmethod
    def _preprocess_positions(
        cls,
        df_positions: pd.DataFrame,
        *,
        spot_spread_bps: float = 1.0,
        recompute_greeks: bool = True,
    ) -> pd.DataFrame:
        df_positions_cp = df_positions.copy()
        start, end = df_positions_cp["date"].min(), df_positions_cp["date"].max()
        tickers = df_positions_cp["ticker"].unique().tolist()
        df_options = OptionLoader.load_data(start, end, process_kwargs={"ticker": tickers})
        df_rates = USRatesLoader.load_data(start, end)
        df_options = compute_forward(df_options, df_rates)
        half_spread = max(float(spot_spread_bps), 0.0) / 10000.0 / 2.0
        df_spot = (
            df_options.groupby(["date", "ticker"], group_keys=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "option_id": x["ticker"].iloc[0],
                        "spot": x["spot"].iloc[0],
                        "bid": x["spot"].iloc[0] * (1.0 - half_spread),
                        "ask": x["spot"].iloc[0] * (1.0 + half_spread),
                        "mid": x["spot"].iloc[0],
                        "strike": x["spot"].iloc[0],
                        "call_put": "C",
                        "implied_volatility": 0.0,
                        "day_to_expiration": 1,
                        "risk_free_rate": x["risk_free_rate"].iloc[0] if "risk_free_rate" in x.columns else 0.0,
                        "delta": 1,
                        "gamma": 0.0,
                        "vega": 0.0,
                        "theta": 0.0,
                    }
                )
            )
            .reset_index()
        )
        df_options_spot = pd.concat([df_options, df_spot], ignore_index=True)
        df_positions_extended = df_positions_cp.merge(df_options_spot, how="left", on=["ticker", "option_id", "date"])
        df_positions_extended = df_positions_extended[(df_positions_extended["date"] <= df_positions_extended["expiration"]) | df_positions_extended["expiration"].isna()]
        df_positions_extended = ffill_options_data(df_positions_extended)
        if recompute_greeks:
            df_positions_extended["dataset_delta"] = df_positions_extended["delta"]
            df_positions_extended["dataset_gamma"] = df_positions_extended["gamma"]
            df_positions_extended["dataset_vega"] = df_positions_extended["vega"]
            df_positions_extended["dataset_theta"] = df_positions_extended["theta"]
            df_positions_extended = compute_black_scholes_greeks(df_positions_extended)
            df_positions_extended["delta"] = df_positions_extended["bs_delta"]
            df_positions_extended["gamma"] = df_positions_extended["bs_gamma"]
            df_positions_extended["vega"] = df_positions_extended["bs_vega"]
            df_positions_extended["theta"] = df_positions_extended["bs_theta"]
        return df_positions_extended

    @classmethod
    def apply_tcost(cls, df_positions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df_positions

    @property
    def pnl(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Backtest has not been run yet.")
        return self._df_pnl

    @property
    def nav(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Backtest has not been run yet.")
        return self._df_nav

    @property
    def metainfo(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Backtest has not been run yet.")
        return self._df_metainfo

    @property
    def drifted_positions(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Backtest has not been run yet.")
        return self._df_drifted_positions


class BacktesterBidAskFromData(StrategyBacktester):
    @classmethod
    def apply_tcost(cls, df_positions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_positions_cp = df_positions.copy()
        trade_in_filter = df_positions_cp["entry_date"] == df_positions_cp["date"]
        trade_out_filter = df_positions_cp["expiration"] == df_positions_cp["date"]
        short_position_filter = df_positions_cp["weight"] < 0
        long_position_filter = ~short_position_filter
        df_positions_cp["mid"] = np.where(trade_in_filter & short_position_filter, df_positions_cp["bid"], df_positions_cp["mid"])
        df_positions_cp["mid"] = np.where(trade_out_filter & short_position_filter, df_positions_cp["ask"], df_positions_cp["mid"])
        df_positions_cp["mid"] = np.where(trade_in_filter & long_position_filter, df_positions_cp["ask"], df_positions_cp["mid"])
        df_positions_cp["mid"] = np.where(trade_out_filter & long_position_filter, df_positions_cp["bid"], df_positions_cp["mid"])
        return df_positions_cp
