SHORT_1M_STRADDLE = [
    {
        "day_to_expiry_target": 28,
        "strike_target": -0.5,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -0.5,
        "leg_name": "Short ATM Put 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 28,
        "strike_target": 0.5,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -0.5,
        "leg_name": "Short ATM Call 1M",
        "rebal_week_day": [2],
    },
]

SHORT_1W_STRANGLE_20D = [
    {
        "day_to_expiry_target": 7,
        "strike_target": -0.2,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -0.5,
        "leg_name": "Short 20D Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 0.2,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -0.5,
        "leg_name": "Short 20D Call 1W",
        "rebal_week_day": [2],
    },
]
