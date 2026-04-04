# Realized Volatility Timing Project

This repository contains a self-contained volatility timing project built around one baseline strategy:

- `SHORT_1M_STRADDLE`

The core idea is to improve a standard short-volatility carry strategy using a model-based volatility forecast:

1. estimate future realized volatility with a Heston-inspired Kalman filter,
2. compare that forecast to market implied volatility,
3. turn the spread into a timing signal,
4. use that signal to scale a short-vol carry strategy.

## Repository layout

- `data/`
  Local option and rates data.
- `src/data/`
  Data loaders for options and US rates.
- `src/models/`
  Heston/Kalman volatility model, Black-Scholes greek recomputation, and GARCH benchmark utilities.
- `src/strategy/`
  Strategy definitions, option selection, and trade generation.
- `src/signal/`
  Signal construction and mapping from forecast spread to trade allocation.
- `src/backtest/`
  Bid/ask-aware backtester with NAV, cash, delta hedge handling, and PnL decomposition.
- `notebooks/final_project.ipynb`
  Final notebook used for analysis and presentation.
- `RAPPORT_KERROUM_VU/`
  Project report accompanying the notebook and implementation.

## Data used

The project uses two data sources:

- option data
  - prices: `bid`, `ask`, `mid`
  - contract fields: `spot`, `strike`, `expiration`, `call_put`, `ticker`
  - market features: `implied_volatility`
- US rates data
  - yield curve points used to interpolate the risk-free rate

Important convention:

- `implied_volatility` comes directly from the dataset
- Black-Scholes greeks are recomputed inside the backtester for a more coherent PnL decomposition

## Current strategy

The active strategy definition lives in:

- `strategies.py`

It is:

- short 1M ATM put
- short 1M ATM call

implemented as a short ATM straddle with weekly entry scheduling.

## Project pipeline

### 1. Load market data

Option and rate data are loaded from local files.

### 2. Build option trades

The strategy is converted into actual option positions by selecting the closest contracts by:

- target maturity
- target delta or moneyness

### 3. Add delta hedge

Trades can be transformed into a delta-hedged book through an explicit hedge leg.

### 4. Estimate future volatility

The Kalman/Heston model is run on the spot series to produce:

- filtered volatility
- forecast volatility

### 5. Build the signal

The timing signal is:

- `vol_signal = IV_reference - forecast_sigma_hat`

This measures whether implied volatility looks rich or cheap relative to the model forecast.

### 6. Backtest

The backtester handles:

- bid/ask execution
- transaction costs
- cash account and NAV
- hedge propagation
- greek-based PnL decomposition

## Main files

- `volatility_project/src/models/heston_kalman.py`
- `volatility_project/src/models/black_scholes_greeks.py`
- `volatility_project/src/signal/vol_signal.py`
- `volatility_project/src/backtest/backtester.py`
- `volatility_project/src/strategy/option_trade.py`
- `volatility_project/notebooks/final_project.ipynb`
- `volatility_project/rapport/`

## Running the project

Install dependencies from:

- `volatility_project/requirements.txt`

Then run the notebook:

- `volatility_project/notebooks/final_project.ipynb`


