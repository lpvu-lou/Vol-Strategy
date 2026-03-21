# Realized Volatility Timing Project

Standalone final project inspired by the Dauphine volatility trading framework.

Structure:
- `data/`: local options and rates data
- `src/`: self-contained project code
- `notebooks/final_project.ipynb`: final analysis notebook

Main features:
- simplified option/rates loaders
- option selection by delta or moneyness
- trade generation with optional delta hedge
- cash/NAV/PnL backtester with bid/ask transaction costs
- Heston-inspired Kalman filter for latent variance estimation
- implied-vs-realized volatility signal
- dynamic allocation overlay on a baseline vol carry strategy

Example:

```python
from src.strategy.strategies import SHORT_1M_STRADDLE
from src.strategy.option_trade import DeltaHedgedOptionTrade
from src.backtest.backtester import BacktesterBidAskFromData
```
