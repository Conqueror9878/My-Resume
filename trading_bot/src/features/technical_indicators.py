"""
Technical Indicators - Comprehensive library of trading indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical analysis indicators for trading.
    All methods are static and work with pandas DataFrames/Series.
    """

    # ==================== Trend Indicators ====================

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    @staticmethod
    def dema(series: pd.Series, period: int) -> pd.Series:
        """Double Exponential Moving Average"""
        ema1 = TechnicalIndicators.ema(series, period)
        ema2 = TechnicalIndicators.ema(ema1, period)
        return 2 * ema1 - ema2

    @staticmethod
    def tema(series: pd.Series, period: int) -> pd.Series:
        """Triple Exponential Moving Average"""
        ema1 = TechnicalIndicators.ema(series, period)
        ema2 = TechnicalIndicators.ema(ema1, period)
        ema3 = TechnicalIndicators.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD - Moving Average Convergence Divergence

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index

        Returns:
            Tuple of (adx, plus_di, minus_di)
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend Indicator

        Returns:
            Tuple of (supertrend, direction)
        """
        atr = TechnicalIndicators.atr(high, low, close, period)
        hl2 = (high + low) / 2

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        for i in range(period, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        return supertrend, direction

    # ==================== Momentum Indicators ====================

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator

        Returns:
            Tuple of (k_line, d_line)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_line = k_line.rolling(window=d_period).mean()
        return k_line, d_line

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        return (typical_price - sma_tp) / (0.015 * mean_deviation)

    @staticmethod
    def roc(series: pd.Series, period: int = 12) -> pd.Series:
        """Rate of Change"""
        return ((series - series.shift(period)) / series.shift(period)) * 100

    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """Momentum"""
        return series - series.shift(period)

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        positive_flow = pd.Series(0.0, index=close.index)
        negative_flow = pd.Series(0.0, index=close.index)

        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = raw_money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = raw_money_flow.iloc[i]

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        money_ratio = positive_mf / negative_mf
        return 100 - (100 / (1 + money_ratio))

    # ==================== Volatility Indicators ====================

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return middle, upper, lower

    @staticmethod
    def keltner_channel(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_mult: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channel

        Returns:
            Tuple of (middle, upper, lower)
        """
        middle = TechnicalIndicators.ema(close, period)
        atr = TechnicalIndicators.atr(high, low, close, period)
        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)
        return middle, upper, lower

    @staticmethod
    def donchian_channel(
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channel

        Returns:
            Tuple of (upper, middle, lower)
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower

    @staticmethod
    def historical_volatility(
        series: pd.Series,
        period: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """Historical Volatility"""
        log_returns = np.log(series / series.shift(1))
        volatility = log_returns.rolling(window=period).std()
        if annualize:
            volatility = volatility * np.sqrt(252)  # Annualized
        return volatility

    # ==================== Volume Indicators ====================

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        direction = np.where(close > close.shift(), 1,
                           np.where(close < close.shift(), -1, 0))
        return (volume * direction).cumsum()

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def accumulation_distribution(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        return (clv * volume).cumsum()

    @staticmethod
    def chaikin_money_flow(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Chaikin Money Flow"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        return (clv * volume).rolling(period).sum() / volume.rolling(period).sum()

    @staticmethod
    def volume_profile(
        close: pd.Series,
        volume: pd.Series,
        bins: int = 20
    ) -> pd.DataFrame:
        """
        Volume Profile - Volume distributed by price level

        Returns:
            DataFrame with price levels and corresponding volume
        """
        price_min, price_max = close.min(), close.max()
        bin_edges = np.linspace(price_min, price_max, bins + 1)

        volume_at_price = []
        for i in range(len(bin_edges) - 1):
            mask = (close >= bin_edges[i]) & (close < bin_edges[i+1])
            vol = volume[mask].sum()
            volume_at_price.append({
                'price_low': bin_edges[i],
                'price_high': bin_edges[i+1],
                'price_mid': (bin_edges[i] + bin_edges[i+1]) / 2,
                'volume': vol
            })

        return pd.DataFrame(volume_at_price)

    # ==================== Pattern Recognition ====================

    @staticmethod
    def pivot_points(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> dict:
        """
        Calculate pivot points

        Returns:
            Dict with pivot, support and resistance levels
        """
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        r1 = 2 * pivot - low.iloc[-1]
        r2 = pivot + (high.iloc[-1] - low.iloc[-1])
        r3 = r1 + (high.iloc[-1] - low.iloc[-1])
        s1 = 2 * pivot - high.iloc[-1]
        s2 = pivot - (high.iloc[-1] - low.iloc[-1])
        s3 = s1 - (high.iloc[-1] - low.iloc[-1])

        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }

    @staticmethod
    def fibonacci_retracement(
        high: float,
        low: float
    ) -> dict:
        """
        Calculate Fibonacci retracement levels

        Returns:
            Dict with Fibonacci levels
        """
        diff = high - low
        return {
            '0.0': high,
            '0.236': high - (diff * 0.236),
            '0.382': high - (diff * 0.382),
            '0.5': high - (diff * 0.5),
            '0.618': high - (diff * 0.618),
            '0.786': high - (diff * 0.786),
            '1.0': low
        }

    @staticmethod
    def ichimoku_cloud(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52
    ) -> dict:
        """
        Ichimoku Cloud

        Returns:
            Dict with all Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)

        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
