"""
Crypto Data Provider - Binance, Coinbase, and other exchanges via CCXT
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

try:
    import ccxt.async_support as ccxt
except ImportError:
    ccxt = None

from config import TimeFrame, Market, config
from .base_provider import BaseDataProvider


logger = logging.getLogger(__name__)


TIMEFRAME_MAP = {
    TimeFrame.M1: '1m',
    TimeFrame.M5: '5m',
    TimeFrame.M15: '15m',
    TimeFrame.M30: '30m',
    TimeFrame.H1: '1h',
    TimeFrame.H4: '4h',
    TimeFrame.D1: '1d',
    TimeFrame.W1: '1w',
}


class CryptoDataProvider(BaseDataProvider):
    """Cryptocurrency data provider using CCXT"""

    def __init__(self, exchange: str = 'binance'):
        super().__init__(Market.CRYPTO)
        self.exchange_name = exchange
        self.exchange = None

    async def connect(self) -> bool:
        """Connect to crypto exchange"""
        if ccxt is None:
            logger.error("ccxt library not installed. Run: pip install ccxt")
            return False

        try:
            exchange_class = getattr(ccxt, self.exchange_name)

            exchange_config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            }

            # Add API keys if available
            if config.api.binance_api_key and self.exchange_name == 'binance':
                exchange_config['apiKey'] = config.api.binance_api_key
                exchange_config['secret'] = config.api.binance_secret

            self.exchange = exchange_class(exchange_config)
            await self.exchange.load_markets()
            self._connected = True
            logger.info(f"Connected to {self.exchange_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_name}: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange"""
        if self.exchange:
            await self.exchange.close()
            self._connected = False
            logger.info(f"Disconnected from {self.exchange_name}")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        tf_str = TIMEFRAME_MAP.get(timeframe, '1h')
        since = int(start.timestamp() * 1000) if start else None

        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=tf_str,
                since=since,
                limit=limit
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Filter by end date if specified
            if end:
                df = df[df.index <= end]

            return self._validate_ohlcv(df)

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker"""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'last': ticker.get('last'),
                'volume': ticker.get('baseVolume'),
                'quote_volume': ticker.get('quoteVolume'),
                'change_24h': ticker.get('percentage'),
                'high_24h': ticker.get('high'),
                'low_24h': ticker.get('low'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    async def fetch_order_book(
        self,
        symbol: str,
        depth: int = 20
    ) -> Dict[str, List]:
        """Fetch order book"""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit=depth)
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise

    async def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs"""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        return list(self.exchange.markets.keys())

    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol trading info"""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        market = self.exchange.market(symbol)
        return {
            'symbol': symbol,
            'base': market.get('base'),
            'quote': market.get('quote'),
            'min_quantity': market.get('limits', {}).get('amount', {}).get('min'),
            'max_quantity': market.get('limits', {}).get('amount', {}).get('max'),
            'min_notional': market.get('limits', {}).get('cost', {}).get('min'),
            'tick_size': market.get('precision', {}).get('price'),
            'lot_size': market.get('precision', {}).get('amount'),
            'maker_fee': market.get('maker'),
            'taker_fee': market.get('taker'),
        }

    async def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Fetch funding rate for perpetual futures"""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = await self.exchange.fetch_funding_rate(symbol)
                return {
                    'symbol': symbol,
                    'funding_rate': funding.get('fundingRate'),
                    'next_funding_time': funding.get('fundingTimestamp'),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.warning(f"Funding rate not available for {symbol}: {e}")
        return None

    async def fetch_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch recent trades"""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        try:
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(t['timestamp'], unit='ms'),
                'price': t['price'],
                'amount': t['amount'],
                'side': t['side'],
                'cost': t['cost']
            } for t in trades])
            return df.set_index('timestamp')
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise


class MultiExchangeCryptoProvider:
    """Aggregator for multiple crypto exchanges"""

    def __init__(self, exchanges: List[str] = None):
        self.exchanges = exchanges or ['binance', 'coinbase']
        self.providers: Dict[str, CryptoDataProvider] = {}

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all exchanges"""
        results = {}
        for exchange in self.exchanges:
            provider = CryptoDataProvider(exchange)
            results[exchange] = await provider.connect()
            if results[exchange]:
                self.providers[exchange] = provider
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges"""
        for provider in self.providers.values():
            await provider.disconnect()

    async def get_best_price(self, symbol: str) -> Dict:
        """Get best bid/ask across all exchanges"""
        best = {'bid': 0, 'ask': float('inf'), 'bid_exchange': None, 'ask_exchange': None}

        for name, provider in self.providers.items():
            try:
                ticker = await provider.fetch_ticker(symbol)
                if ticker['bid'] and ticker['bid'] > best['bid']:
                    best['bid'] = ticker['bid']
                    best['bid_exchange'] = name
                if ticker['ask'] and ticker['ask'] < best['ask']:
                    best['ask'] = ticker['ask']
                    best['ask_exchange'] = name
            except Exception:
                continue

        return best
