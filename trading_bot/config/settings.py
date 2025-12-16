"""
Institutional Trading Bot - Configuration Settings
Supports: Crypto, Forex, Stocks
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Market(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"


class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class APIConfig:
    """API credentials configuration"""
    # Crypto - Binance/Coinbase
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_secret: str = os.getenv("BINANCE_SECRET", "")
    coinbase_api_key: str = os.getenv("COINBASE_API_KEY", "")
    coinbase_secret: str = os.getenv("COINBASE_SECRET", "")

    # Forex - OANDA/Interactive Brokers
    oanda_api_key: str = os.getenv("OANDA_API_KEY", "")
    oanda_account_id: str = os.getenv("OANDA_ACCOUNT_ID", "")
    ib_host: str = os.getenv("IB_HOST", "127.0.0.1")
    ib_port: int = int(os.getenv("IB_PORT", "7497"))

    # Stocks - Alpaca/Interactive Brokers
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret: str = os.getenv("ALPACA_SECRET", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


@dataclass
class RiskConfig:
    """Risk management parameters"""
    max_position_size: float = 0.02  # 2% of portfolio per position
    max_portfolio_risk: float = 0.06  # 6% total portfolio at risk
    max_drawdown: float = 0.15  # 15% max drawdown before halt
    max_daily_loss: float = 0.03  # 3% max daily loss
    max_correlation: float = 0.7  # Max correlation between positions
    var_confidence: float = 0.99  # VaR confidence level
    max_leverage: Dict[str, float] = field(default_factory=lambda: {
        "crypto": 3.0,
        "forex": 30.0,
        "stocks": 4.0
    })
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0


@dataclass
class TradingConfig:
    """Trading parameters"""
    markets: List[Market] = field(default_factory=lambda: [Market.CRYPTO, Market.FOREX, Market.STOCKS])

    # Crypto pairs
    crypto_symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"
    ])

    # Forex pairs
    forex_symbols: List[str] = field(default_factory=lambda: [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "USD/CAD"
    ])

    # Stock symbols
    stock_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ"
    ])

    # Timeframes for analysis
    primary_timeframe: TimeFrame = TimeFrame.H1
    secondary_timeframes: List[TimeFrame] = field(default_factory=lambda: [
        TimeFrame.M15, TimeFrame.H4, TimeFrame.D1
    ])

    # Execution settings
    slippage_tolerance: float = 0.001  # 0.1%
    min_volume_filter: float = 1000000  # Minimum daily volume
    order_timeout: int = 30  # seconds


@dataclass
class ModelConfig:
    """ML model configuration"""
    model_type: str = "ensemble"  # ensemble, lstm, transformer, xgboost, rl

    # Feature settings
    lookback_periods: int = 100
    prediction_horizon: int = 5

    # Training settings
    train_test_split: float = 0.8
    validation_split: float = 0.1
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "lstm": 0.3,
        "xgboost": 0.3,
        "transformer": 0.2,
        "technical": 0.2
    })

    # Reinforcement Learning settings
    rl_algorithm: str = "PPO"  # PPO, A2C, DQN
    rl_episodes: int = 1000
    rl_gamma: float = 0.99
    rl_epsilon: float = 0.1


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "fixed"  # fixed, variable, volume_based
    slippage_bps: float = 5.0  # basis points


@dataclass
class Config:
    """Master configuration"""
    api: APIConfig = field(default_factory=APIConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"

    # Mode
    paper_trading: bool = True
    debug: bool = False


# Global config instance
config = Config()
