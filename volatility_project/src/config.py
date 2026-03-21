from pathlib import Path

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25

TENOR_TO_PERIOD = {
    "1 Mo": 1 / 12,
    "2 Mo": 1 / 6,
    "3 Mo": 1 / 4,
    "4 Mo": 1 / 3,
    "6 Mo": 0.5,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
