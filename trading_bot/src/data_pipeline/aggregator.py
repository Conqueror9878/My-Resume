"""
Data Aggregator - Unified interface for all market data
"""

import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import logging

from config import Market, TimeFrame, config
from .base_provider import BaseDataProvider
from .crypto_provider import CryptoDataProvider
from .forex_provider import OANDADataProvider
from .stock_provider import AlpacaDataProvider, YahooFinanceProvider


logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Unified data aggregator for all markets.
    Provides a single interface to fetch data from crypto, forex, and stock markets.
    """

    def __init__(self):
        self.providers: Dict[Market, BaseDataProvider] = {}
        self._initialized = False

    async def initialize(
        self,
        markets: Optional[List[Market]] = None,
        use_free_providers: bool = False
    ) -> Dict[Market, bool]:
        """
        Initialize data providers for specified markets.

        Args:
            markets: List of markets to initialize. Defaults to all configured markets.
            use_free_providers: Use free data providers (Yahoo Finance instead of Alpaca)

        Returns:
            Dict mapping market to connection success status
        """
        if markets is None:
            markets = config.trading.markets

        results = {}

        for market in markets:
            try:
                if market == Market.CRYPTO:
                    provider = CryptoDataProvider('binance')
                elif market == Market.FOREX:
                    provider = OANDADataProvider(practice=config.paper_trading)
                elif market == Market.STOCKS:
                    if use_free_providers:
                        provider = YahooFinanceProvider()
                    else:
                        provider = AlpacaDataProvider()
                else:
                    logger.warning(f"Unknown market: {market}")
                    continue

                success = await provider.connect()
                results[market] = success

                if success:
                    self.providers[market] = provider
                    logger.info(f"Initialized {market.value} provider")
                else:
                    logger.error(f"Failed to initialize {market.value} provider")

            except Exception as e:
                logger.error(f"Error initializing {market.value}: {e}")
                results[market] = False

        self._initialized = len(self.providers) > 0
        return results

    async def shutdown(self) -> None:
        """Disconnect all providers"""
        for provider in self.providers.values():
            await provider.disconnect()
        self.providers.clear()
        self._initialized = False
        logger.info("All providers disconnected")

    def get_provider(self, market: Market) -> Optional[BaseDataProvider]:
        """Get provider for specific market"""
        return self.providers.get(market)

    def _get_market_for_symbol(self, symbol: str) -> Optional[Market]:
        """Determine market type from symbol format"""
        if '/' in symbol:
            # Could be crypto (BTC/USDT) or forex (EUR/USD)
            base, quote = symbol.split('/')
            forex_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'}
            if base in forex_currencies and quote in forex_currencies:
                return Market.FOREX
            return Market.CRYPTO
        else:
            # Stock symbol
            return Market.STOCKS

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        market: Optional[Market] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for any symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT', 'EUR/USD', 'AAPL')
            timeframe: Candle timeframe
            market: Market type. Auto-detected if not specified.
            start: Start datetime
            end: End datetime
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        if market is None:
            market = self._get_market_for_symbol(symbol)

        provider = self.providers.get(market)
        if not provider:
            raise ValueError(f"No provider available for market: {market}")

        return await provider.fetch_ohlcv(symbol, timeframe, start, end, limit)

    async def fetch_ticker(
        self,
        symbol: str,
        market: Optional[Market] = None
    ) -> Dict:
        """Fetch current ticker for any symbol"""
        if market is None:
            market = self._get_market_for_symbol(symbol)

        provider = self.providers.get(market)
        if not provider:
            raise ValueError(f"No provider available for market: {market}")

        return await provider.fetch_ticker(symbol)

    async def fetch_multiple_ohlcv(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            timeframe: Candle timeframe
            **kwargs: Additional arguments passed to fetch_ohlcv

        Returns:
            Dict mapping symbol to DataFrame
        """
        tasks = [
            self.fetch_ohlcv(symbol, timeframe, **kwargs)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {symbol}: {result}")
            else:
                data[symbol] = result

        return data

    async def fetch_all_configured(
        self,
        timeframe: TimeFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all configured symbols.

        Returns:
            Dict mapping symbol to DataFrame
        """
        all_symbols = []

        if Market.CRYPTO in self.providers:
            all_symbols.extend(config.trading.crypto_symbols)

        if Market.FOREX in self.providers:
            all_symbols.extend(config.trading.forex_symbols)

        if Market.STOCKS in self.providers:
            all_symbols.extend(config.trading.stock_symbols)

        return await self.fetch_multiple_ohlcv(all_symbols, timeframe, **kwargs)

    async def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: Optional[List[TimeFrame]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes. Defaults to configured timeframes.

        Returns:
            Dict mapping timeframe string to DataFrame
        """
        if timeframes is None:
            timeframes = [config.trading.primary_timeframe] + config.trading.secondary_timeframes

        market = self._get_market_for_symbol(symbol)
        provider = self.providers.get(market)

        if not provider:
            raise ValueError(f"No provider available for market: {market}")

        return await provider.fetch_multi_timeframe(symbol, timeframes)

    async def get_correlation_matrix(
        self,
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.D1,
        periods: int = 100
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for symbols.

        Args:
            symbols: List of symbols
            timeframe: Timeframe for analysis
            periods: Number of periods

        Returns:
            Correlation matrix as DataFrame
        """
        data = await self.fetch_multiple_ohlcv(symbols, timeframe, limit=periods)

        # Extract close prices
        closes = pd.DataFrame()
        for symbol, df in data.items():
            if not df.empty:
                closes[symbol] = df['close'].pct_change()

        return closes.corr()

    def get_connected_markets(self) -> List[Market]:
        """Get list of connected markets"""
        return list(self.providers.keys())

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# Global instance
data_aggregator = DataAggregator()
