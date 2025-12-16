# Institutional AI Trading Bot

A comprehensive, institutional-grade AI trading bot supporting **Crypto**, **Forex**, and **Stocks** markets with machine learning-based signal generation, risk management, and backtesting capabilities.

## Features

- **Multi-Market Support**: Trade across Crypto (Binance, Coinbase), Forex (OANDA), and Stocks (Alpaca)
- **ML-Powered Signals**: LSTM, Transformer, XGBoost, LightGBM, and Ensemble models
- **Reinforcement Learning**: PPO/A2C/DQN agents for adaptive trading
- **Risk Management**: Position sizing, VaR, drawdown limits, correlation monitoring
- **Backtesting Engine**: Walk-forward optimization, comprehensive metrics
- **Paper Trading**: Test strategies without risking real capital

## Project Structure

```
trading_bot/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── src/
│   ├── data_pipeline/        # Market data providers
│   │   ├── base_provider.py
│   │   ├── crypto_provider.py
│   │   ├── forex_provider.py
│   │   ├── stock_provider.py
│   │   └── aggregator.py
│   ├── features/             # Feature engineering
│   │   ├── technical_indicators.py
│   │   └── feature_engineer.py
│   ├── models/               # ML models
│   │   ├── base_model.py
│   │   ├── lstm_model.py
│   │   ├── xgboost_model.py
│   │   ├── ensemble_model.py
│   │   └── rl_model.py
│   ├── backtesting/          # Backtesting engine
│   │   └── engine.py
│   ├── risk/                 # Risk management
│   │   └── risk_manager.py
│   ├── execution/            # Order execution
│   │   └── order_manager.py
│   ├── strategies/           # Trading strategies
│   │   └── base_strategy.py
│   ├── utils/                # Utilities
│   │   └── logger.py
│   └── bot.py               # Main bot orchestrator
├── data/                     # Data storage
├── models/                   # Saved models
├── logs/                     # Log files
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### 1. Clone and Setup

```bash
cd trading_bot
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API credentials
```

## Quick Start

### 1. Paper Trading with Momentum Strategy

```python
import asyncio
from src.bot import TradingBot
from src.strategies import MomentumStrategy
from config import Market

async def main():
    # Create strategy
    strategy = MomentumStrategy(
        rsi_oversold=30,
        rsi_overbought=70,
        trend_period=50
    )

    # Create bot
    bot = TradingBot(strategy=strategy, paper_trading=True)

    # Initialize
    await bot.initialize(
        markets=[Market.CRYPTO],
        symbols={Market.CRYPTO: ['BTC/USDT', 'ETH/USDT']}
    )

    # Start trading
    await bot.start(interval_seconds=60)

asyncio.run(main())
```

### 2. Backtesting

```python
import asyncio
from src.bot import TradingBot
from src.strategies import TrendFollowingStrategy
from config import Market

async def main():
    strategy = TrendFollowingStrategy(fast_ma=10, slow_ma=30)
    bot = TradingBot(strategy=strategy)

    await bot.initialize(markets=[Market.CRYPTO])

    result = await bot.backtest(
        symbol='BTC/USDT',
        start_date='2023-01-01',
        end_date='2024-01-01',
        market=Market.CRYPTO
    )

asyncio.run(main())
```

### 3. Training ML Model

```python
import asyncio
from src.bot import TradingBot
from src.strategies import MLStrategy
from config import Market

async def main():
    ml_strategy = MLStrategy(threshold=0.6)
    bot = TradingBot(strategy=ml_strategy)

    await bot.initialize(markets=[Market.STOCKS])

    # Train model
    model = await bot.train_model(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        market=Market.STOCKS
    )

    # Backtest with trained model
    result = await bot.backtest('AAPL', '2023-01-01', '2024-01-01')

asyncio.run(main())
```

## Strategies

### Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `MomentumStrategy` | RSI + MACD with trend filter |
| `MeanReversionStrategy` | Bollinger Bands + RSI |
| `TrendFollowingStrategy` | MA crossover |
| `MLStrategy` | ML model predictions |
| `EnsembleStrategy` | Combines multiple strategies |

### Custom Strategy

```python
from src.strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("my_strategy")

    def generate_signal(self, data, **kwargs):
        # Your logic here
        # Return: 1 (buy), -1 (sell), 0 (hold)
        return 0
```

## Configuration

Edit `config/settings.py` or use environment variables:

```python
# Risk Settings
max_position_size = 0.02      # 2% per position
max_portfolio_risk = 0.06     # 6% total risk
max_drawdown = 0.15           # 15% max drawdown
max_daily_loss = 0.03         # 3% daily loss limit

# Model Settings
lookback_periods = 100
prediction_horizon = 5
ensemble_weights = {
    'lstm': 0.3,
    'xgboost': 0.3,
    'transformer': 0.2,
    'technical': 0.2
}
```

## API Integration

### Supported Exchanges/Brokers

| Market | Provider | Paper Trading | Live Trading |
|--------|----------|---------------|--------------|
| Crypto | Binance | ✅ | ✅ |
| Crypto | Coinbase | ✅ | ✅ |
| Forex | OANDA | ✅ | ✅ |
| Stocks | Alpaca | ✅ | ✅ |
| Stocks | Yahoo Finance | ✅ (data only) | ❌ |

## Risk Management

The bot includes institutional-grade risk management:

- **Position Sizing**: Kelly criterion, volatility-adjusted, fixed fraction
- **Stop Loss**: ATR-based, percentage, support/resistance
- **Portfolio Risk**: VaR, correlation monitoring, exposure limits
- **Circuit Breakers**: Max drawdown, daily loss limits

## Performance Metrics

Backtest reports include:

- Total/Annual Return
- Sharpe/Sortino/Calmar Ratios
- Maximum Drawdown
- Win Rate, Profit Factor
- Trade Statistics

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Formatting

```bash
black src/
isort src/
flake8 src/
```

## Disclaimer

This software is for educational purposes only. Trading financial instruments involves substantial risk of loss. Past performance is not indicative of future results. Always paper trade first and never risk more than you can afford to lose.

## License

MIT License
