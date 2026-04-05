from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import DAYS_PER_YEAR

# onvert days to expiration to annualised time, floored at 1 day to avoid division by zero
def _safe_tau(days_to_expiration: pd.Series | np.ndarray) -> np.ndarray:
    tau = np.asarray(days_to_expiration, dtype=float) / DAYS_PER_YEAR
    return np.maximum(tau, 1.0 / DAYS_PER_YEAR)

# Clip implied volatility to a small positive value to prevent numerical issues
def _safe_sigma(implied_volatility: pd.Series | np.ndarray) -> np.ndarray:
    sigma = np.asarray(implied_volatility, dtype=float)
    return np.maximum(sigma, 1e-8)

# Compute d1 and d2 for the Black-Scholes formula, with safeguards against division by zero
def _d1_d2(
    spot: np.ndarray,
    strike: np.ndarray,
    rate: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(np.maximum(spot, 1e-12) / np.maximum(strike, 1e-12)) + (rate + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2

# Compute Black-Scholes Greeks for a DataFrame of option data, handling both options and delta hedging legs
def compute_black_scholes_greeks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Separate option legs from delta hedging legs based on the 'leg_name' column
    option_mask = out["leg_name"] != "DELTA_HEDGING"
    if not option_mask.any():
        return out

    opt = out.loc[option_mask].copy()
    spot = np.asarray(opt["spot"], dtype=float)
    strike = np.asarray(opt["strike"], dtype=float)
    rate = np.asarray(opt.get("risk_free_rate", 0.0), dtype=float)
    sigma = _safe_sigma(opt["implied_volatility"])
    tau = _safe_tau(opt["day_to_expiration"])

    # Precompute sqrt(tau) and d1, d2 for efficiency, and calculate the probability density function of d1
    sqrt_tau = np.sqrt(tau)
    d1, d2 = _d1_d2(spot, strike, rate, sigma, tau)
    pdf_d1 = norm.pdf(d1)

    is_call = opt["call_put"].eq("C").to_numpy()
    delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = pdf_d1 / (np.maximum(spot, 1e-12) * sigma * sqrt_tau)
    vega = np.maximum(spot, 1e-12) * pdf_d1 * sqrt_tau

    call_theta = (
        -(spot * pdf_d1 * sigma) / (2.0 * sqrt_tau)
        - rate * strike * np.exp(-rate * tau) * norm.cdf(d2)
    ) / DAYS_PER_YEAR
    put_theta = (
        -(spot * pdf_d1 * sigma) / (2.0 * sqrt_tau)
        + rate * strike * np.exp(-rate * tau) * norm.cdf(-d2)
    ) / DAYS_PER_YEAR
    theta = np.where(is_call, call_theta, put_theta)

    # Assign computed Greeks to the output DataFrame for option legs
    out.loc[option_mask, "bs_gamma"] = gamma
    out.loc[option_mask, "bs_vega"] = vega
    out.loc[option_mask, "bs_theta"] = theta

    # For delta hedging legs, set Greeks to their expected values (delta = 1 for the underlying, gamma and vega = 0)    
    hedge_mask = ~option_mask
    out.loc[hedge_mask, "bs_delta"] = 1.0
    out.loc[hedge_mask, "bs_gamma"] = 0.0
    out.loc[hedge_mask, "bs_vega"] = 0.0
    out.loc[hedge_mask, "bs_theta"] = 0.0
    return out
