from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


@dataclass
class HestonKalmanEstimator:
    kappa: float = 3.0
    theta: float = 0.04
    xi: float = 0.35
    rho: float = -0.7
    observation_noise: float = 5e-7
    process_noise: float = 5e-2
    initial_variance: float = 0.04
    initial_covariance: float = 1.0
    variance_floor: float = 1e-8
    auto_calibrate: bool = True
    theta_window: int = 63
    initial_window: int = 21

    def __post_init__(self) -> None:
        self._state_history: pd.DataFrame | None = None

    @property
    def dt(self) -> float:
        return 1.0 / TRADING_DAYS_PER_YEAR

    def fit(self, spot: pd.Series) -> "HestonKalmanEstimator":
        self._state_history = self.filter(spot)
        return self

    def filter(self, spot: pd.Series) -> pd.DataFrame:
        df = self._prepare_spot_frame(spot)
        if df.empty:
            return df.assign(v_hat=pd.Series(dtype=float), sigma_hat=pd.Series(dtype=float))

        self._calibrate_from_returns(df["log_return"])
        state_mean = np.log(max(self.initial_variance, self.variance_floor))
        state_cov = self.initial_covariance
        filtered_rows: list[dict[str, float | pd.Timestamp]] = []

        for row in df.itertuples(index=False):
            observed_sq_return = float(row.log_return) ** 2
            predicted_variance = self._transition_variance(np.exp(state_mean))
            predicted_mean = np.log(predicted_variance)
            transition_jacobian = self._transition_jacobian(np.exp(state_mean), predicted_variance)
            process_variance = self._process_variance(np.exp(state_mean))
            predicted_cov = transition_jacobian * state_cov * transition_jacobian + process_variance
            observation_mean = predicted_variance * self.dt
            observation_jacobian = predicted_variance * self.dt
            innovation_variance = observation_jacobian * predicted_cov * observation_jacobian + self.observation_noise
            kalman_gain = 0.0 if innovation_variance <= 0 else (predicted_cov * observation_jacobian) / innovation_variance
            innovation = observed_sq_return - observation_mean
            state_mean = predicted_mean + kalman_gain * innovation
            state_cov = max((1 - kalman_gain * observation_jacobian) * predicted_cov, self.variance_floor)
            filtered_variance = max(np.exp(state_mean), self.variance_floor)
            filtered_rows.append(
                {
                    "date": row.date,
                    "spot": float(row.spot),
                    "log_return": float(row.log_return),
                    "v_hat": filtered_variance,
                    "sigma_hat": np.sqrt(filtered_variance),
                }
            )
        return pd.DataFrame(filtered_rows)

    def fit_transform(self, spot: pd.Series) -> pd.DataFrame:
        return self.fit(spot)._state_history.copy()

    def forecast_average_variance(self, variance: pd.Series | np.ndarray | float, horizon_days: int) -> pd.Series | np.ndarray | float:
        """Forecast annualized average variance over the option horizon."""

        horizon = horizon_days / TRADING_DAYS_PER_YEAR
        if horizon <= 0:
            return variance
        kappa = max(self.kappa, 1e-12)
        decay = np.exp(-kappa * horizon)
        loading = (1.0 - decay) / (kappa * horizon)
        return self.theta + (variance - self.theta) * loading

    def add_horizon_forecast(self, filtered_df: pd.DataFrame, horizon_days: int, prefix: str = "horizon") -> pd.DataFrame:
        """Append horizon-matched variance and volatility forecasts."""

        out = filtered_df.copy()
        out[f"{prefix}_days"] = horizon_days
        out[f"{prefix}_variance_hat"] = self.forecast_average_variance(out["v_hat"], horizon_days=horizon_days)
        out[f"{prefix}_sigma_hat"] = np.sqrt(out[f"{prefix}_variance_hat"].clip(lower=self.variance_floor))
        return out

    def _prepare_spot_frame(self, spot: pd.Series) -> pd.DataFrame:
        spot_series = pd.Series(spot).dropna().astype(float)
        df = spot_series.rename("spot").to_frame()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df["log_return"] = np.log(df["spot"]).diff().fillna(0.0)
        return df.reset_index(names="date")

    def _calibrate_from_returns(self, log_returns: pd.Series) -> None:
        if not self.auto_calibrate:
            return

        returns = pd.Series(log_returns).dropna().astype(float)
        if returns.empty:
            return

        annualized_sq_returns = returns.pow(2) * TRADING_DAYS_PER_YEAR
        min_theta_obs = max(5, self.theta_window // 3)
        min_initial_obs = max(5, self.initial_window // 3)

        theta_estimates = annualized_sq_returns.rolling(self.theta_window, min_periods=min_theta_obs).mean().dropna()
        initial_estimates = annualized_sq_returns.rolling(self.initial_window, min_periods=min_initial_obs).mean().dropna()

        if not theta_estimates.empty:
            theta = float(theta_estimates.median())
            self.theta = max(theta, self.variance_floor)
        if not initial_estimates.empty:
            initial_variance = float(initial_estimates.iloc[0])
            self.initial_variance = max(initial_variance, self.variance_floor)

    def _transition_variance(self, variance: float) -> float:
        return max(variance + self.kappa * (self.theta - variance) * self.dt, self.variance_floor)

    def _transition_jacobian(self, variance: float, transitioned_variance: float) -> float:
        slope = 1 - self.kappa * self.dt
        return slope * variance / max(transitioned_variance, self.variance_floor)

    def _process_variance(self, variance: float) -> float:
        heston_component = (self.xi**2) * max(variance, self.variance_floor) * self.dt
        return max((heston_component / max(variance, self.variance_floor)) + self.process_noise, self.variance_floor)
