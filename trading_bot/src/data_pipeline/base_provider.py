"""
Base Data Provider - Abstract interface for all market data sources
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import logging

from config import TimeFrame, Market


logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """Abstract base class for all data providers"""

    def __init__(self, market: Market):
        self.market = market
        self.cache: Dict[str, pd.DataFrame] = {}
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data

        Returns DataFrame with columns:
        - timestamp (index)
        - open
        - high
        - low
        - close
        - volume
        """
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data

        Returns dict with:
        - bid
        - ask
        - last
        - volume
        - timestamp
        """
        pass

    @abstractmethod
    async def fetch_order_book(
        self,
        symbol: str,
        depth: int = 20
    ) -> Dict[str, List]:
        """
        Fetch order book data

        Returns dict with:
        - bids: List of [price, quantity]
        - asks: List of [price, quantity]
        """
        pass

    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols"""
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get symbol metadata

        Returns dict with:
        - min_quantity
        - max_quantity
        - tick_size
        - lot_size
        - margin_required (if applicable)
        """
        pass

    async def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[TimeFrame],
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        result = {}
        for tf in timeframes:
            try:
                result[tf.value] = await self.fetch_ohlcv(symbol, tf, limit=limit)
            except Exception as e:
                logger.error(f"Failed to fetch {tf.value} for {symbol}: {e}")
        return result

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]

        # Sort by timestamp
        df = df.sort_index()

        # Forward fill small gaps (up to 3 periods)
        df = df.ffill(limit=3)

        # Drop remaining NaN rows
        df = df.dropna()

        return df

    def _cache_key(self, symbol: str, timeframe: TimeFrame) -> str:
        """Generate cache key"""
        return f"{self.market.value}_{symbol}_{timeframe.value}"

    def get_cached(
        self,
        symbol: str,
        timeframe: TimeFrame
    ) -> Optional[pd.DataFrame]:
        """Get cached data if available"""
        key = self._cache_key(symbol, timeframe)
        return self.cache.get(key)

    def set_cache(
        self,
        symbol: str,
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> None:
        """Cache data"""
        key = self._cache_key(symbol, timeframe)
        self.cache[key] = data

    @property
    def is_connected(self) -> bool:
        return self._connected
