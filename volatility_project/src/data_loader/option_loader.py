from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.utils.helpers import check_is_true


class DataLoader(ABC):
    _EXTENSION_TO_LOADER = {
        "parquet": pd.read_parquet,
        "csv": pd.read_csv,
        "xlsx": pd.read_excel,
    }

    @classmethod
    def load_data(
        cls,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        load_kwargs: Optional[dict] = None,
        process_kwargs: Optional[dict] = None,
        extra_fields_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        file_path = str(cls._get_path())
        min_date, max_date = cls._get_valid_date_range()
        start_date = start_date or min_date
        end_date = end_date or max_date
        check_is_true(start_date <= end_date, "start_date must be before end_date")
        check_is_true(start_date >= min_date and end_date <= max_date, f"Data only available between {min_date.date()} and {max_date.date()}")
        extension = file_path.split(".")[-1]
        check_is_true(extension in cls._EXTENSION_TO_LOADER, f"Unsupported file extension: {extension}")
        df = cls._EXTENSION_TO_LOADER[extension](file_path, **(load_kwargs or {}))
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df = cls._process_loaded_data(df, **(process_kwargs or {}))
        df = cls._add_extra_fields(df, **(extra_fields_kwargs or {}))
        return df[df["date"].between(start_date, end_date)]

    @classmethod
    @abstractmethod
    def _get_path(cls) -> Path:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        raise NotImplementedError

    @classmethod
    def _process_loaded_data(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    @classmethod
    def _add_extra_fields(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df


class OptionLoader(DataLoader):
    @classmethod
    def _get_path(cls) -> Path:
        return DATA_DIR / "optiondb_2016_2023.parquet"

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        return (datetime(2016, 1, 2), datetime(2023, 12, 30))

    @classmethod
    def _process_loaded_data(cls, df: pd.DataFrame, *, ticker: str | Sequence[str], **kwargs) -> pd.DataFrame:
        tickers = [ticker] if isinstance(ticker, str) else list(ticker)
        df = df[df["ticker"].isin(tickers)].copy()
        df["expiration"] = pd.to_datetime(df["expiration"], format="mixed")
        df["volume"] = df["volume"].fillna(0)
        return cls._compute_final_payoff(df)

    @classmethod
    def _add_extra_fields(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df["expiration"] = pd.to_datetime(df["expiration"], format="mixed")
        df["day_to_expiration"] = (df["expiration"] - df["date"]).dt.days
        df["moneyness"] = df["strike"] / df["spot"]
        return df

    @staticmethod
    def _compute_final_payoff(df_option: pd.DataFrame) -> pd.DataFrame:
        df_option = df_option.copy()
        expiring_filter = df_option["date"] == df_option["expiration"]
        expiring_calls_filter = expiring_filter & (df_option["call_put"] == "C")
        expiring_puts_filter = expiring_filter & (df_option["call_put"] == "P")
        call_payoff = (df_option["spot"] - df_option["strike"]).clip(lower=0)
        put_payoff = (df_option["strike"] - df_option["spot"]).clip(lower=0)
        for col in ("mid", "bid", "ask"):
            df_option[col] = np.where(
                expiring_calls_filter,
                call_payoff,
                np.where(expiring_puts_filter, put_payoff, df_option[col]),
            )
        return df_option


def extract_spot_from_options(df_options: pd.DataFrame) -> pd.DataFrame:
    return (
        df_options[["date", "spot"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
