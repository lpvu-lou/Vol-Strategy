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
