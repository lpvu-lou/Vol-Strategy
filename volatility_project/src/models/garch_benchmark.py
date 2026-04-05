from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import TRADING_DAYS_PER_YEAR

# Immutable container for fitted GARCH(1,1) parameters
@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float

# Recursively compute the conditional variance path for a GARCH(1,1) process
def _garch_variance_path(returns: np.ndarray, params: GarchParams, initial_variance: float) -> np.ndarray:
    variances = np.empty_like(returns, dtype=float)
    prev_variance = float(max(initial_variance, 1e-12))
    for idx, ret in enumerate(returns):
        current_variance = params.omega + params.alpha * (ret ** 2) + params.beta * prev_variance
        current_variance = max(current_variance, 1e-12)
        variances[idx] = current_variance
        prev_variance = current_variance
    return variances

# Evaluate the negative log-likelihood of the GARCH(1,1) model given parameters and return data
def _neg_loglik_garch(raw_params: np.ndarray, returns: np.ndarray, initial_variance: float) -> float:
    omega, alpha, beta = raw_params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12
    params = GarchParams(omega=float(omega), alpha=float(alpha), beta=float(beta))
    variances = _garch_variance_path(returns, params, initial_variance)
    return float(0.5 * np.sum(np.log(variances) + (returns ** 2) / variances))

# Fit a GARCH(1,1) model to the provided return series and return the estimated parameters
def fit_garch11(returns: pd.Series) -> GarchParams:
    clean_returns = pd.Series(returns).dropna().astype(float)
    if clean_returns.empty:
        raise ValueError("Cannot fit GARCH(1,1) on an empty return series.")

    returns_np = clean_returns.to_numpy()
    initial_variance = float(max(clean_returns.var(ddof=1), 1e-8))

    # Start with a reasonable initial guess and enforce constraints to ensure stationarity and positivity of parameters
    initial_guess = np.array(
        [
            initial_variance * 0.05,
            0.05,
            0.90,
        ],
        dtype=float,
    )

    # Bounds ensure omega is positive and not excessively large, alpha and beta are non-negative, and their sum is less than 1 for stationarity
    bounds = [
        (1e-12, max(initial_variance * 10.0, 1e-6)),
        (1e-6, 0.35),
        (1e-6, 0.999),
    ]

    # Hard constraint to ensure alpha + beta < 1 for stationarity of the GARCH process
    constraints = (
        {
            "type": "ineq",
            "fun": lambda x: 0.999 - x[1] - x[2],
        },
    )

    result = minimize(
        _neg_loglik_garch,
        x0=initial_guess,
        args=(returns_np, initial_variance),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )

    if result.success:
        omega, alpha, beta = result.x
    else:
        # Stable fallback close to market-standard persistence if optimization struggles.
        omega, alpha, beta = initial_variance * 0.05, 0.05, 0.90

    return GarchParams(omega=float(omega), alpha=float(alpha), beta=float(beta))

# Generate a rolling GARCH(1,1) annualised volatility forecast
# At each date, fit the GARCH(1,1) model to the most recent 'window' returns and forecast the next 'horizon_days' variances
def forecast_garch11_volatility(
    returns: pd.Series,
    *,
    window: int = 252,
    horizon_days: int = 21,
    recalibration_frequency: int = 21,
) -> pd.DataFrame:
    clean_returns = pd.Series(returns).dropna().astype(float)
    clean_returns.index = pd.to_datetime(clean_returns.index)
    if len(clean_returns) < window:
        raise ValueError("Not enough observations to compute a rolling GARCH(1,1) forecast.")

    # Re-fit only at recalibration steps; reuse cached params otherwise
    records: list[dict[str, float | pd.Timestamp]] = []
    cached_params: GarchParams | None = None

    # Run variance path through the window to get the latest conditional variance
    for end_idx in range(window - 1, len(clean_returns)):
        window_returns = clean_returns.iloc[end_idx - window + 1 : end_idx + 1]
        if cached_params is None or (end_idx - (window - 1)) % recalibration_frequency == 0:
            cached_params = fit_garch11(window_returns)

        # One-step ahead forecast of variance using the last return and last conditional variance from the window
        assert cached_params is not None
        window_variance = float(max(window_returns.var(ddof=1), 1e-8))
        conditional_variances = _garch_variance_path(window_returns.to_numpy(), cached_params, window_variance)
        last_variance = float(conditional_variances[-1])
        last_return_sq = float(window_returns.iloc[-1] ** 2)

        # Forecast the variance path for the next 'horizon_days' using the GARCH(1,1) recursion
        next_variance = cached_params.omega + cached_params.alpha * last_return_sq + cached_params.beta * last_variance
        forecast_path = [max(next_variance, 1e-12)]
        for _ in range(1, horizon_days):
            next_variance = cached_params.omega + (cached_params.alpha + cached_params.beta) * forecast_path[-1]
            forecast_path.append(max(next_variance, 1e-12))

        # Average the forecasted variances over the horizon and annualize to get the volatility estimate
        annualized_sigma = float(np.sqrt(np.mean(forecast_path) * TRADING_DAYS_PER_YEAR))
        current_date = clean_returns.index[end_idx]
        records.append(
            {
                "date": current_date,
                "log_return": float(clean_returns.iloc[end_idx]),
                "garch_sigma_hat": annualized_sigma,
                "garch_omega": cached_params.omega,
                "garch_alpha": cached_params.alpha,
                "garch_beta": cached_params.beta,
            }
        )

    return pd.DataFrame.from_records(records)
