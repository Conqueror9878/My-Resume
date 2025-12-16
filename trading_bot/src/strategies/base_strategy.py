"""
Base Strategy - Abstract interface for trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging

from src.features import FeatureEngineer, TechnicalIndicators
from config import config


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.feature_engineer = FeatureEngineer()
        self.indicators = TechnicalIndicators
        self.position = 0  # -1, 0, 1

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> int:
        """
        Generate trading signal from data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Signal: 1 (buy), -1 (sell), 0 (hold)
        """
        pass

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()
        close = df['close']
        high = df['high']
        low = df['low']

        # Moving averages
        df['sma_20'] = self.indicators.sma(close, 20)
        df['sma_50'] = self.indicators.sma(close, 50)
        df['ema_12'] = self.indicators.ema(close, 12)
        df['ema_26'] = self.indicators.ema(close, 26)

        # RSI
        df['rsi'] = self.indicators.rsi(close, 14)

        # MACD
        macd, signal, hist = self.indicators.macd(close)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # Bollinger Bands
        _, upper, lower = self.indicators.bollinger_bands(close)
        df['bb_upper'] = upper
        df['bb_lower'] = lower

        # ATR
        df['atr'] = self.indicators.atr(high, low, close)

        return df


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        trend_period: int = 50
    ):
        super().__init__("momentum")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_period = trend_period

    def generate_signal(self, data: pd.DataFrame, **kwargs) -> int:
        """Generate signal based on momentum indicators"""
        if len(data) < self.trend_period:
            return 0

        df = self.calculate_indicators(data)

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Trend filter
        trend_up = current['close'] > current[f'sma_{self.trend_period}']
        trend_down = current['close'] < current[f'sma_{self.trend_period}']

        # RSI signals
        rsi_buy = prev['rsi'] < self.rsi_oversold and current['rsi'] > self.rsi_oversold
        rsi_sell = prev['rsi'] > self.rsi_overbought and current['rsi'] < self.rsi_overbought

        # MACD crossover
        macd_buy = prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']
        macd_sell = prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']

        # Generate signal
        if trend_up and (rsi_buy or macd_buy):
            return 1
        elif trend_down and (rsi_sell or macd_sell):
            return -1

        return 0


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14
    ):
        super().__init__("mean_reversion")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

    def generate_signal(self, data: pd.DataFrame, **kwargs) -> int:
        """Generate signal based on mean reversion"""
        if len(data) < self.bb_period + 5:
            return 0

        df = self.calculate_indicators(data)

        current = df.iloc[-1]
        close = current['close']

        # Bollinger Band signals
        at_lower_band = close <= current['bb_lower']
        at_upper_band = close >= current['bb_upper']

        # RSI confirmation
        rsi_oversold = current['rsi'] < 30
        rsi_overbought = current['rsi'] > 70

        # Buy at lower band with RSI oversold
        if at_lower_band and rsi_oversold:
            return 1

        # Sell at upper band with RSI overbought
        if at_upper_band and rsi_overbought:
            return -1

        return 0


class TrendFollowingStrategy(BaseStrategy):
    """Trend following trading strategy"""

    def __init__(
        self,
        fast_ma: int = 10,
        slow_ma: int = 30,
        atr_multiplier: float = 2.0
    ):
        super().__init__("trend_following")
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.atr_multiplier = atr_multiplier

    def generate_signal(self, data: pd.DataFrame, **kwargs) -> int:
        """Generate signal based on trend following"""
        if len(data) < self.slow_ma + 5:
            return 0

        df = data.copy()
        close = df['close']

        # Calculate MAs
        df['fast_ma'] = self.indicators.ema(close, self.fast_ma)
        df['slow_ma'] = self.indicators.ema(close, self.slow_ma)

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # MA crossover
        golden_cross = prev['fast_ma'] < prev['slow_ma'] and current['fast_ma'] > current['slow_ma']
        death_cross = prev['fast_ma'] > prev['slow_ma'] and current['fast_ma'] < current['slow_ma']

        if golden_cross:
            return 1
        elif death_cross:
            return -1

        return 0


class MLStrategy(BaseStrategy):
    """Machine learning based strategy"""

    def __init__(self, model=None, threshold: float = 0.6):
        super().__init__("ml_strategy")
        self.model = model
        self.threshold = threshold
        self.lookback = config.model.lookback_periods

    def set_model(self, model):
        """Set the ML model"""
        self.model = model

    def generate_signal(self, data: pd.DataFrame, **kwargs) -> int:
        """Generate signal using ML model"""
        if self.model is None or not self.model.is_trained:
            logger.warning("ML model not available")
            return 0

        if len(data) < self.lookback + 10:
            return 0

        try:
            # Generate features
            features_df = self.feature_engineer.generate_features(data, include_target=False)

            if features_df.empty:
                return 0

            # Get latest features
            X, _ = self.feature_engineer.get_feature_matrix(
                features_df.iloc[-1:],
                normalize=True
            )

            # For sequence models
            if hasattr(self.model, 'create_sequences'):
                X_full, _ = self.feature_engineer.get_feature_matrix(
                    features_df.iloc[-self.lookback:],
                    normalize=True
                )
                X = X_full.reshape(1, self.lookback, -1)

            # Get signal from model
            return self.model.get_signal(X, self.threshold)

        except Exception as e:
            logger.error(f"ML signal generation error: {e}")
            return 0


class EnsembleStrategy(BaseStrategy):
    """Combines multiple strategies with voting"""

    def __init__(self, strategies: List[BaseStrategy] = None, weights: List[float] = None):
        super().__init__("ensemble")
        self.strategies = strategies or []
        self.weights = weights or [1.0] * len(self.strategies)

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the ensemble"""
        self.strategies.append(strategy)
        self.weights.append(weight)

    def generate_signal(self, data: pd.DataFrame, **kwargs) -> int:
        """Generate signal by combining strategy votes"""
        if not self.strategies:
            return 0

        weighted_sum = 0
        total_weight = 0

        for strategy, weight in zip(self.strategies, self.weights):
            try:
                signal = strategy.generate_signal(data, **kwargs)
                weighted_sum += signal * weight
                total_weight += weight
            except Exception as e:
                logger.error(f"Strategy {strategy.name} error: {e}")

        if total_weight == 0:
            return 0

        # Threshold for ensemble signal
        avg_signal = weighted_sum / total_weight

        if avg_signal > 0.5:
            return 1
        elif avg_signal < -0.5:
            return -1

        return 0
