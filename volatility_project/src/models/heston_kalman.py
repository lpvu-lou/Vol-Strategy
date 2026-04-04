from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import TRADING_DAYS_PER_YEAR


@dataclass
class HestonKalmanEstimator:
    kappa: float = 1.2
    theta: float = 0.04
    xi: float = 0.6
    rho: float = -0.5
    observation_noise: float = 1e-6
    process_noise: float = 1e-6
    initial_variance: float = 0.04
    initial_covariance: float = 0.02
    variance_floor: float = 1e-8
    variance_ceiling: float = 4.0
    drift: float = 0.0
    rolling_window: int = 126
    recalibration_frequency: int = 21
    auto_calibrate: bool = True
    optimize_rho: bool = True
    calibration_smoothness: float = 1e-3
    boundary_penalty: float = 1e-4
    feller_penalty: float = 1e-2
    boundary_buffer: float = 1e-3
    alpha: float = 0.1
    beta: float = 2.0
    ukf_kappa: float = 0.0
    innovation_clip_std: float = 6.0

    def __post_init__(self) -> None:
        self._state_history: pd.DataFrame | None = None
        self._parameter_history: pd.DataFrame | None = None
        self._calibration_history: pd.DataFrame | None = None

    @property
    def dt(self) -> float:
        return 1.0 / TRADING_DAYS_PER_YEAR

    def fit(self, spot: pd.Series) -> "HestonKalmanEstimator":
        self._state_history = self.filter(spot)
        return self

    def fit_transform(self, spot: pd.Series) -> pd.DataFrame:
        return self.fit(spot)._state_history.copy()

    def filter(self, spot: pd.Series) -> pd.DataFrame:
        df = self._prepare_spot_frame(spot)
        if df.empty:
            return df.assign(v_hat=pd.Series(dtype=float), sigma_hat=pd.Series(dtype=float))

        current_params = self._current_parameter_vector()
        state_mean = self._bounded_variance(self.initial_variance)
        state_covariance = max(self.initial_covariance, self.variance_floor)
        recalibration_frequency = max(int(self.recalibration_frequency), 1)
        filtered_rows: list[dict[str, float | pd.Timestamp]] = []
        parameter_rows: list[dict[str, float | pd.Timestamp]] = []
        calibration_rows: list[dict[str, float | pd.Timestamp]] = []

        for idx, row in enumerate(df.itertuples(index=False)):
            if self.auto_calibrate and idx >= self.rolling_window and (idx == self.rolling_window or idx % recalibration_frequency == 0):
                window_returns = df.iloc[idx - self.rolling_window : idx]["log_return"]
                window_initial_variance = max(
                    float(np.nanmean(window_returns.to_numpy() ** 2) * TRADING_DAYS_PER_YEAR),
                    self.variance_floor,
                )
                current_params, diagnostics = self._fit_window_parameters(
                    window_returns,
                    current_params,
                    initial_variance=window_initial_variance,
                )
                calibration_rows.append({"date": row.date, **diagnostics})

            state_mean, state_covariance, loglikelihood = self._ukf_step(
                observation=float(row.log_return),
                state_mean=state_mean,
                state_covariance=state_covariance,
                params=current_params,
            )
            filtered_rows.append(
                {
                    "date": row.date,
                    "spot": float(row.spot),
                    "log_return": float(row.log_return),
                    "v_hat": state_mean,
                    "sigma_hat": np.sqrt(max(state_mean, self.variance_floor)),
                    "kappa_used": current_params["kappa"],
                    "theta_used": current_params["theta"],
                    "xi_used": current_params["xi"],
                    "rho_used": current_params["rho"],
                    "process_noise_used": current_params["process_noise"],
                    "observation_noise_used": current_params["observation_noise"],
                    "loglikelihood": loglikelihood,
                }
            )
            parameter_rows.append({"date": row.date, **current_params})

        self._parameter_history = pd.DataFrame(parameter_rows)
        self._calibration_history = pd.DataFrame(calibration_rows)
        return pd.DataFrame(filtered_rows)

    def forecast_average_variance(
        self,
        variance: pd.Series | np.ndarray | float,
        horizon_days: int,
        *,
        kappa: pd.Series | np.ndarray | float | None = None,
        theta: pd.Series | np.ndarray | float | None = None,
    ) -> pd.Series | np.ndarray | float:
        horizon = horizon_days / TRADING_DAYS_PER_YEAR
        if horizon <= 0:
            return variance
        kappa_value = np.maximum(self.kappa if kappa is None else kappa, 1e-12)
        theta_value = self.theta if theta is None else theta
        decay = np.exp(-kappa_value * horizon)
        loading = (1.0 - decay) / (kappa_value * horizon)
        return theta_value + (variance - theta_value) * loading

    def add_horizon_forecast(self, filtered_df: pd.DataFrame, horizon_days: int, prefix: str = "horizon") -> pd.DataFrame:
        out = filtered_df.copy()
        kappa = out["kappa_used"] if "kappa_used" in out.columns else self.kappa
        theta = out["theta_used"] if "theta_used" in out.columns else self.theta
        out[f"{prefix}_days"] = horizon_days
        out[f"{prefix}_variance_hat"] = self.forecast_average_variance(out["v_hat"], horizon_days=horizon_days, kappa=kappa, theta=theta)
        out[f"{prefix}_sigma_hat"] = np.sqrt(out[f"{prefix}_variance_hat"].clip(lower=self.variance_floor))
        return out

    def _prepare_spot_frame(self, spot: pd.Series) -> pd.DataFrame:
        spot_series = pd.Series(spot).dropna().astype(float)
        df = spot_series.rename("spot").to_frame()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df["log_return"] = np.log(df["spot"]).diff()
        df = df.dropna(subset=["log_return"])
        return df.reset_index(names="date")

    def _current_parameter_vector(self) -> dict[str, float]:
        return {
            "kappa": max(float(self.kappa), 1e-6),
            "theta": max(float(self.theta), self.variance_floor),
            "xi": max(float(self.xi), 1e-6),
            "rho": float(np.clip(self.rho, -0.999, 0.999)),
            "process_noise": max(float(self.process_noise), self.variance_floor),
            "observation_noise": max(float(self.observation_noise), self.variance_floor),
        }

    def _fit_window_parameters(
        self,
        returns_window: pd.Series,
        current_params: dict[str, float],
        *,
        initial_variance: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        clean_returns = returns_window.dropna()
        if clean_returns.empty:
            return current_params, self._empty_calibration_diagnostics()

        initial_guess = np.array(
            [
                np.log(current_params["kappa"]),
                np.log(current_params["theta"]),
                np.log(current_params["xi"]),
                np.arctanh(np.clip(current_params["rho"], -0.95, 0.95)),
            ],
            dtype=float,
        )
        bounds = [
            (np.log(0.05), np.log(6.0)),
            (np.log(1e-4), np.log(0.50)),
            (np.log(0.05), np.log(1.0)),
            (-1.83, 1.83),
        ]

        def objective(raw_params: np.ndarray) -> float:
            params = {
                "kappa": float(np.exp(raw_params[0])),
                "theta": float(np.exp(raw_params[1])),
                "xi": float(np.exp(raw_params[2])),
                "rho": float(np.tanh(raw_params[3])) if self.optimize_rho else current_params["rho"],
                "process_noise": current_params["process_noise"],
                "observation_noise": current_params["observation_noise"],
            }
            neg_loglikelihood = self._negative_loglikelihood(clean_returns.to_numpy(), params=params, initial_variance=initial_variance)
            smoothness_penalty = self.calibration_smoothness * float(np.sum((raw_params - initial_guess) ** 2))
            boundary_penalty = self.boundary_penalty * self._boundary_penalty(raw_params, bounds)
            feller_penalty = self.feller_penalty * self._feller_violation_penalty(params)
            total = neg_loglikelihood + smoothness_penalty + boundary_penalty + feller_penalty
            return total

        result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            diagnostics = self._empty_calibration_diagnostics()
            diagnostics["objective_success"] = 0.0
            return current_params, diagnostics

        clipped = self._clip_inside_bounds(result.x, bounds)
        calibrated_params = {
            "kappa": float(np.exp(clipped[0])),
            "theta": float(np.exp(clipped[1])),
            "xi": float(np.exp(clipped[2])),
            "rho": float(np.tanh(clipped[3])) if self.optimize_rho else current_params["rho"],
            "process_noise": current_params["process_noise"],
            "observation_noise": current_params["observation_noise"],
        }
        diagnostics = self._objective_components(
            clipped,
            bounds=bounds,
            params=calibrated_params,
            initial_guess=initial_guess,
            returns=clean_returns.to_numpy(),
            initial_variance=initial_variance,
        )
        diagnostics["objective_success"] = 1.0
        diagnostics["used_inner_clip"] = float(not np.allclose(clipped, result.x))
        return calibrated_params, diagnostics

    def _negative_loglikelihood(self, returns: np.ndarray, params: dict[str, float], initial_variance: float) -> float:
        state_mean = self._bounded_variance(initial_variance)
        state_covariance = max(self.initial_covariance, self.variance_floor)
        total = 0.0

        for obs in returns:
            state_mean, state_covariance, loglikelihood = self._ukf_step(
                observation=float(obs),
                state_mean=state_mean,
                state_covariance=state_covariance,
                params=params,
            )
            total -= loglikelihood
        if not np.isfinite(total):
            return 1e12
        return float(total)

    def _ukf_step(
        self,
        *,
        observation: float,
        state_mean: float,
        state_covariance: float,
        params: dict[str, float],
    ) -> tuple[float, float, float]:
        sigma_points, wm, wc = self._augmented_sigma_points(state_mean, state_covariance)
        propagated_states = np.array(
            [self._state_transition(v, z_state, params) for v, z_state, _ in sigma_points],
            dtype=float,
        )
        predicted_state_mean = float(np.sum(wm * propagated_states))
        predicted_state_mean = self._bounded_variance(predicted_state_mean)
        predicted_state_covariance = float(np.sum(wc * (propagated_states - predicted_state_mean) ** 2))
        predicted_state_covariance += params["process_noise"]
        predicted_state_covariance = max(predicted_state_covariance, self.variance_floor)

        measurement_sigma_points, wm, wc = self._augmented_sigma_points(predicted_state_mean, predicted_state_covariance)
        propagated_measurements = np.array(
            [self._measurement_function(v, z_state, z_obs, params) for v, z_state, z_obs in measurement_sigma_points],
            dtype=float,
        )
        propagated_state_points = np.array([max(point[0], self.variance_floor) for point in measurement_sigma_points], dtype=float)
        predicted_measurement_mean = float(np.sum(wm * propagated_measurements))
        predicted_measurement_covariance = float(np.sum(wc * (propagated_measurements - predicted_measurement_mean) ** 2))
        predicted_measurement_covariance += params["observation_noise"]
        predicted_measurement_covariance = max(predicted_measurement_covariance, self.variance_floor)
        cross_covariance = float(
            np.sum(wc * (propagated_state_points - predicted_state_mean) * (propagated_measurements - predicted_measurement_mean))
        )
        kalman_gain = cross_covariance / predicted_measurement_covariance
        innovation = observation - predicted_measurement_mean
        innovation_std = np.sqrt(predicted_measurement_covariance)
        clipped_innovation = float(
            np.clip(
                innovation,
                -self.innovation_clip_std * innovation_std,
                self.innovation_clip_std * innovation_std,
            )
        )
        updated_state_mean = self._bounded_variance(predicted_state_mean + kalman_gain * clipped_innovation)
        updated_state_covariance = max(
            predicted_state_covariance - kalman_gain * predicted_measurement_covariance * kalman_gain,
            self.variance_floor,
        )
        # Use the same innovation in the likelihood and in the state update.
        loglikelihood = -0.5 * (
            np.log(2.0 * np.pi * predicted_measurement_covariance)
            + (clipped_innovation**2) / predicted_measurement_covariance
        )
        return updated_state_mean, updated_state_covariance, float(loglikelihood)

    def _augmented_sigma_points(self, state_mean: float, state_covariance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        augmented_mean = np.array([state_mean, 0.0, 0.0], dtype=float)
        augmented_covariance = np.diag([max(state_covariance, self.variance_floor), 1.0, 1.0])
        n_dim = augmented_mean.size
        lambda_ = self.alpha**2 * (n_dim + self.ukf_kappa) - n_dim
        scale = n_dim + lambda_
        root = np.linalg.cholesky(scale * augmented_covariance)
        sigma_points = [augmented_mean]
        for i in range(n_dim):
            sigma_points.append(augmented_mean + root[:, i])
            sigma_points.append(augmented_mean - root[:, i])
        sigma_points_array = np.asarray(sigma_points, dtype=float)
        wm = np.full(2 * n_dim + 1, 1.0 / (2.0 * scale), dtype=float)
        wc = wm.copy()
        wm[0] = lambda_ / scale
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        return sigma_points_array, wm, wc

    def _state_transition(self, variance: float, z_state: float, params: dict[str, float]) -> float:
        variance = self._bounded_variance(variance)
        next_variance = variance + params["kappa"] * (params["theta"] - variance) * self.dt
        next_variance += params["xi"] * np.sqrt(variance * self.dt) * z_state
        return self._bounded_variance(next_variance)

    def _measurement_function(self, variance: float, z_state: float, z_obs: float, params: dict[str, float]) -> float:
        variance = self._bounded_variance(variance)
        correlated_shock = params["rho"] * z_state + np.sqrt(max(1.0 - params["rho"] ** 2, 1e-10)) * z_obs
        return (self.drift - 0.5 * variance) * self.dt + np.sqrt(variance * self.dt) * correlated_shock

    def _bounded_variance(self, value: float) -> float:
        return float(np.clip(value, self.variance_floor, self.variance_ceiling))

    def _boundary_penalty(self, raw_params: np.ndarray, bounds: list[tuple[float, float]]) -> float:
        penalty = 0.0
        for value, (lower, upper) in zip(raw_params, bounds):
            width = max(upper - lower, 1e-12)
            lower_gap = max(value - lower, 1e-12)
            upper_gap = max(upper - value, 1e-12)
            target_gap = max(self.boundary_buffer * width, 1e-12)
            if lower_gap < target_gap:
                penalty += ((target_gap / lower_gap) - 1.0) ** 2
            if upper_gap < target_gap:
                penalty += ((target_gap / upper_gap) - 1.0) ** 2
        return float(penalty)

    def _clip_inside_bounds(self, raw_params: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
        clipped = []
        for value, (lower, upper) in zip(raw_params, bounds):
            width = upper - lower
            eps = self.boundary_buffer * width
            clipped.append(np.clip(value, lower + eps, upper - eps))
        return np.asarray(clipped, dtype=float)

    def _feller_violation_penalty(self, params: dict[str, float]) -> float:
        violation = max(params["xi"] ** 2 - 2.0 * params["kappa"] * params["theta"], 0.0)
        scale = max(params["xi"] ** 2, 1e-12)
        return float((violation / scale) ** 2)

    def _objective_components(
        self,
        raw_params: np.ndarray,
        *,
        bounds: list[tuple[float, float]],
        params: dict[str, float],
        initial_guess: np.ndarray,
        returns: np.ndarray,
        initial_variance: float,
    ) -> dict[str, float]:
        neg_loglikelihood = self._negative_loglikelihood(returns, params=params, initial_variance=initial_variance)
        smoothness_penalty = self.calibration_smoothness * float(np.sum((raw_params - initial_guess) ** 2))
        boundary_penalty = self.boundary_penalty * self._boundary_penalty(raw_params, bounds)
        feller_penalty = self.feller_penalty * self._feller_violation_penalty(params)
        return {
            "objective_total": float(neg_loglikelihood + smoothness_penalty + boundary_penalty + feller_penalty),
            "neg_loglikelihood_term": float(neg_loglikelihood),
            "smoothness_penalty_term": float(smoothness_penalty),
            "boundary_penalty_term": float(boundary_penalty),
            "feller_penalty_term": float(feller_penalty),
        }

    def _empty_calibration_diagnostics(self) -> dict[str, float]:
        return {
            "objective_total": np.nan,
            "neg_loglikelihood_term": np.nan,
            "smoothness_penalty_term": np.nan,
            "boundary_penalty_term": np.nan,
            "feller_penalty_term": np.nan,
            "objective_success": np.nan,
            "used_inner_clip": np.nan,
        }
