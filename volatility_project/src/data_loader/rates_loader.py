from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.config import DATA_DIR
from src.data_loader.option_loader import DataLoader


# Concrete implementation of DataLoader for loading and processing US Treasury par yield curve rates
class USRatesLoader(DataLoader):
    # Return the path to the US Treasury par yield curve rates CSV file
    @classmethod
    def _get_path(cls):
        return DATA_DIR / "par-yield-curve-rates-2020-2023.csv"

    # The data file contains rates from Jan 2, 2020 to Dec 30, 2023, so we set these as the valid date range for filtering
    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        return (datetime(2020, 1, 2), datetime(2023, 12, 30))

    # Parse dates, forward-fill missing values, sort by date, and convert rates from percent to decimal
    @classmethod
    def _process_loaded_data(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df = df.ffill().set_index("date").sort_index() / 100
        return df.reset_index()
