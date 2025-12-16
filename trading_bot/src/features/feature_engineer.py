"""
Feature Engineering - Transform raw OHLCV data into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from .technical_indicators import TechnicalIndicators
from config import config


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for trading ML models.
    Generates technical, statistical, and derived features.
    """

    def __init__(self, lookback: int = None):
        self.lookback = lookback or config.model.lookback_periods
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._fitted = False

    def generate_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
        horizon: int = None
    ) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            include_target: Whether to include target variable
            horizon: Prediction horizon for target

        Returns:
            DataFrame with all features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()

        horizon = horizon or config.model.prediction_horizon
        features = df.copy()

        # Generate all feature categories
        features = self._add_price_features(features)
        features = self._add_return_features(features)
        features = self._add_trend_features(features)
        features = self._add_momentum_features(features)
        features = self._add_volatility_features(features)
        features = self._add_volume_features(features)
        features = self._add_pattern_features(features)
        features = self._add_statistical_features(features)
        features = self._add_time_features(features)

        # Add target if requested
        if include_target:
            features = self._add_target(features, horizon)

        # Drop NaN rows from indicator calculations
        features = features.dropna()

        # Store feature names
        self.feature_names = [col for col in features.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'target', 'target_class']]

        return features

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Price ratios
        df['hl_ratio'] = df['high'] / df['low']
        df['co_ratio'] = df['close'] / df['open']
        df['oc_range'] = df['open'] - df['close']
        df['hl_range'] = df['high'] - df['low']

        # Gap
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)

        # Distance from high/low
        df['dist_from_high'] = (df['high'] - df['close']) / df['high']
        df['dist_from_low'] = (df['close'] - df['low']) / df['low']

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features"""
        close = df['close']

        # Simple returns at various horizons
        for period in [1, 2, 3, 5, 10, 20, 50]:
            df[f'return_{period}'] = close.pct_change(period)

        # Log returns
        df['log_return_1'] = np.log(close / close.shift(1))
        df['log_return_5'] = np.log(close / close.shift(5))

        # Cumulative returns
        df['cum_return_5'] = (1 + df['return_1']).rolling(5).apply(np.prod) - 1
        df['cum_return_20'] = (1 + df['return_1']).rolling(20).apply(np.prod) - 1

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        ti = TechnicalIndicators

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ti.sma(close, period)
            df[f'ema_{period}'] = ti.ema(close, period)

            # Price relative to MA
            df[f'close_sma_{period}_ratio'] = close / df[f'sma_{period}']

        # MA crossovers
        df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ema_12_26_cross'] = (df['ema_10'] > df['ema_20']).astype(int)

        # MACD
        macd, signal, hist = ti.macd(close)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['macd_cross'] = (macd > signal).astype(int)

        # ADX
        adx, plus_di, minus_di = ti.adx(high, low, close)
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_cross'] = (plus_di > minus_di).astype(int)

        # Ichimoku
        ichimoku = ti.ichimoku_cloud(high, low, close)
        df['ichimoku_tenkan'] = ichimoku['tenkan_sen']
        df['ichimoku_kijun'] = ichimoku['kijun_sen']
        df['ichimoku_tk_cross'] = (ichimoku['tenkan_sen'] > ichimoku['kijun_sen']).astype(int)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        ti = TechnicalIndicators

        # RSI at different periods
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ti.rsi(close, period)

        # RSI zones
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

        # Stochastic
        k, d = ti.stochastic(high, low, close)
        df['stoch_k'] = k
        df['stoch_d'] = d
        df['stoch_cross'] = (k > d).astype(int)

        # Williams %R
        df['williams_r'] = ti.williams_r(high, low, close)

        # CCI
        df['cci'] = ti.cci(high, low, close)

        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ti.roc(close, period)

        # MFI
        df['mfi'] = ti.mfi(high, low, close, volume)

        # Momentum
        df['momentum_10'] = ti.momentum(close, 10)
        df['momentum_20'] = ti.momentum(close, 20)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        ti = TechnicalIndicators

        # ATR at different periods
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = ti.atr(high, low, close, period)

        # ATR percentage
        df['atr_pct'] = df['atr_14'] / close

        # Bollinger Bands
        for period in [10, 20]:
            middle, upper, lower = ti.bollinger_bands(close, period)
            df[f'bb_middle_{period}'] = middle
            df[f'bb_upper_{period}'] = upper
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
            df[f'bb_position_{period}'] = (close - lower) / (upper - lower)

        # Keltner Channel
        kc_middle, kc_upper, kc_lower = ti.keltner_channel(high, low, close)
        df['kc_position'] = (close - kc_lower) / (kc_upper - kc_lower)

        # Historical volatility
        df['hist_vol_20'] = ti.historical_volatility(close, 20)
        df['hist_vol_60'] = ti.historical_volatility(close, 60)

        # Volatility ratio
        df['vol_ratio'] = df['hist_vol_20'] / df['hist_vol_60']

        # Price range
        df['daily_range'] = (high - low) / close
        df['range_sma_20'] = ti.sma(df['daily_range'], 20)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        ti = TechnicalIndicators

        # Volume moving averages
        for period in [5, 10, 20, 50]:
            df[f'volume_sma_{period}'] = ti.sma(volume, period)
            df[f'volume_ratio_{period}'] = volume / df[f'volume_sma_{period}']

        # OBV
        df['obv'] = ti.obv(close, volume)
        df['obv_sma_20'] = ti.sma(df['obv'], 20)

        # VWAP (intraday)
        df['vwap'] = ti.vwap(high, low, close, volume)
        df['vwap_distance'] = (close - df['vwap']) / df['vwap']

        # Accumulation/Distribution
        df['ad_line'] = ti.accumulation_distribution(high, low, close, volume)

        # Chaikin Money Flow
        df['cmf'] = ti.chaikin_money_flow(high, low, close, volume)

        # Volume momentum
        df['volume_momentum'] = volume.pct_change(5)

        # Price-volume correlation
        df['pv_corr_20'] = close.rolling(20).corr(volume)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']

        # Candle body and wick sizes
        body = abs(c - o)
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        total_range = h - l

        df['body_pct'] = body / total_range
        df['upper_wick_pct'] = upper_wick / total_range
        df['lower_wick_pct'] = lower_wick / total_range

        # Bullish/Bearish candle
        df['is_bullish'] = (c > o).astype(int)

        # Doji (small body)
        df['is_doji'] = (body / total_range < 0.1).astype(int)

        # Hammer (small body at top, long lower wick)
        df['is_hammer'] = ((df['lower_wick_pct'] > 0.6) &
                          (df['body_pct'] < 0.3) &
                          (df['upper_wick_pct'] < 0.1)).astype(int)

        # Engulfing pattern
        prev_body = body.shift(1)
        df['is_bullish_engulfing'] = ((c > o) &
                                       (c.shift(1) < o.shift(1)) &
                                       (body > prev_body)).astype(int)
        df['is_bearish_engulfing'] = ((c < o) &
                                       (c.shift(1) > o.shift(1)) &
                                       (body > prev_body)).astype(int)

        # Higher highs and lower lows
        df['higher_high'] = (h > h.shift(1)).astype(int)
        df['lower_low'] = (l < l.shift(1)).astype(int)
        df['higher_high_streak'] = df['higher_high'].groupby(
            (~df['higher_high'].astype(bool)).cumsum()
        ).cumsum()

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        close = df['close']
        returns = df.get('return_1', close.pct_change())

        for window in [20, 50]:
            # Rolling statistics
            df[f'mean_{window}'] = close.rolling(window).mean()
            df[f'std_{window}'] = close.rolling(window).std()
            df[f'skew_{window}'] = returns.rolling(window).skew()
            df[f'kurt_{window}'] = returns.rolling(window).kurt()

            # Z-score
            df[f'zscore_{window}'] = (close - df[f'mean_{window}']) / df[f'std_{window}']

            # Percentile rank
            df[f'pct_rank_{window}'] = close.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )

        # Autocorrelation
        df['autocorr_1'] = returns.rolling(50).apply(lambda x: x.autocorr(1))
        df['autocorr_5'] = returns.rolling(50).apply(lambda x: x.autocorr(5))

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter

            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            # Is market open indicators (for stocks)
            df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)

        return df

    def _add_target(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add target variable for supervised learning"""
        # Future return
        df['target'] = df['close'].shift(-horizon) / df['close'] - 1

        # Classification target (up/down/neutral)
        threshold = df['target'].std() * 0.5
        df['target_class'] = pd.cut(
            df['target'],
            bins=[-np.inf, -threshold, threshold, np.inf],
            labels=[0, 1, 2]  # 0=down, 1=neutral, 2=up
        )

        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature matrix and target for ML models.

        Args:
            df: DataFrame with features
            feature_cols: Columns to use as features. Uses all if None.
            normalize: Whether to normalize features

        Returns:
            Tuple of (X, y) numpy arrays
        """
        if feature_cols is None:
            feature_cols = self.feature_names

        X = df[feature_cols].values
        y = df['target'].values if 'target' in df.columns else None

        if normalize:
            if not self._fitted:
                X = self.scaler.fit_transform(X)
                self._fitted = True
            else:
                X = self.scaler.transform(X)

        return X, y

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer models.

        Args:
            X: Feature matrix
            y: Target array
            sequence_length: Length of sequences

        Returns:
            Tuple of (X_seq, y_seq) with shape (samples, sequence_length, features)
        """
        sequence_length = sequence_length or self.lookback

        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            if y is not None:
                y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq) if y is not None else None

    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)
