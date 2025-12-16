"""
Stock Data Provider - Alpaca and Yahoo Finance integration
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import aiohttp

from config import TimeFrame, Market, config
from .base_provider import BaseDataProvider


logger = logging.getLogger(__name__)


ALPACA_TIMEFRAME_MAP = {
    TimeFrame.M1: '1Min',
    TimeFrame.M5: '5Min',
    TimeFrame.M15: '15Min',
    TimeFrame.M30: '30Min',
    TimeFrame.H1: '1Hour',
    TimeFrame.H4: '4Hour',
    TimeFrame.D1: '1Day',
    TimeFrame.W1: '1Week',
}


class AlpacaDataProvider(BaseDataProvider):
    """Stock data provider using Alpaca API"""

    DATA_URL = "https://data.alpaca.markets"

    def __init__(self):
        super().__init__(Market.STOCKS)
        self.api_key = config.api.alpaca_api_key
        self.api_secret = config.api.alpaca_secret
        self.base_url = config.api.alpaca_base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API credentials not configured")
            return False

        try:
            self.session = aiohttp.ClientSession(
                headers={
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.api_secret
                }
            )

            # Test connection
            async with self.session.get(
                f"{self.base_url}/v2/account"
            ) as response:
                if response.status == 200:
                    self._connected = True
                    logger.info("Connected to Alpaca")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Alpaca connection failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca"""
        if self.session:
            await self.session.close()
            self._connected = False
            logger.info("Disconnected from Alpaca")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpaca"""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        tf_str = ALPACA_TIMEFRAME_MAP.get(timeframe, '1Hour')

        # Default date range
        if not end:
            end = datetime.now()
        if not start:
            start = end - timedelta(days=365)

        params = {
            'timeframe': tf_str,
            'start': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'limit': limit,
            'adjustment': 'split'  # Adjust for stock splits
        }

        try:
            async with self.session.get(
                f"{self.DATA_URL}/v2/stocks/{symbol}/bars",
                params=params
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Alpaca API error: {error}")

                data = await response.json()
                bars = data.get('bars', [])

                if not bars:
                    return pd.DataFrame()

                records = []
                for bar in bars:
                    records.append({
                        'timestamp': pd.to_datetime(bar['t']),
                        'open': float(bar['o']),
                        'high': float(bar['h']),
                        'low': float(bar['l']),
                        'close': float(bar['c']),
                        'volume': int(bar['v']),
                        'vwap': float(bar.get('vw', 0)),
                        'trade_count': int(bar.get('n', 0))
                    })

                df = pd.DataFrame(records)
                df.set_index('timestamp', inplace=True)
                return self._validate_ohlcv(df)

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current quote"""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Fetch latest quote
            async with self.session.get(
                f"{self.DATA_URL}/v2/stocks/{symbol}/quotes/latest"
            ) as response:
                quote_data = await response.json()

            # Fetch latest trade
            async with self.session.get(
                f"{self.DATA_URL}/v2/stocks/{symbol}/trades/latest"
            ) as response:
                trade_data = await response.json()

            quote = quote_data.get('quote', {})
            trade = trade_data.get('trade', {})

            return {
                'bid': float(quote.get('bp', 0)),
                'ask': float(quote.get('ap', 0)),
                'bid_size': int(quote.get('bs', 0)),
                'ask_size': int(quote.get('as', 0)),
                'last': float(trade.get('p', 0)),
                'last_size': int(trade.get('s', 0)),
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
        """
        Alpaca doesn't provide full order book for free.
        Returns latest quote as single-level book.
        """
        ticker = await self.fetch_ticker(symbol)
        return {
            'bids': [[ticker['bid'], ticker['bid_size']]],
            'asks': [[ticker['ask'], ticker['ask_size']]],
            'timestamp': ticker['timestamp']
        }

    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable stocks"""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            async with self.session.get(
                f"{self.base_url}/v2/assets",
                params={'status': 'active', 'asset_class': 'us_equity'}
            ) as response:
                assets = await response.json()

            return [
                asset['symbol']
                for asset in assets
                if asset.get('tradable', False)
            ]

        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            raise

    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol trading info"""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            async with self.session.get(
                f"{self.base_url}/v2/assets/{symbol}"
            ) as response:
                asset = await response.json()

            return {
                'symbol': symbol,
                'name': asset.get('name'),
                'exchange': asset.get('exchange'),
                'asset_class': asset.get('class'),
                'tradable': asset.get('tradable'),
                'marginable': asset.get('marginable'),
                'shortable': asset.get('shortable'),
                'easy_to_borrow': asset.get('easy_to_borrow'),
                'fractionable': asset.get('fractionable'),
                'min_quantity': 1 if not asset.get('fractionable') else 0.001,
                'tick_size': 0.01
            }

        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            raise

    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            async with self.session.get(
                f"{self.base_url}/v2/account"
            ) as response:
                account = await response.json()

            return {
                'buying_power': float(account.get('buying_power', 0)),
                'cash': float(account.get('cash', 0)),
                'portfolio_value': float(account.get('portfolio_value', 0)),
                'equity': float(account.get('equity', 0)),
                'last_equity': float(account.get('last_equity', 0)),
                'long_market_value': float(account.get('long_market_value', 0)),
                'short_market_value': float(account.get('short_market_value', 0)),
                'initial_margin': float(account.get('initial_margin', 0)),
                'maintenance_margin': float(account.get('maintenance_margin', 0)),
                'daytrading_buying_power': float(account.get('daytrading_buying_power', 0)),
                'pattern_day_trader': account.get('pattern_day_trader', False),
                'currency': account.get('currency', 'USD')
            }

        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise

    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            async with self.session.get(
                f"{self.base_url}/v2/positions"
            ) as response:
                positions = await response.json()

            return [{
                'symbol': pos['symbol'],
                'qty': float(pos['qty']),
                'side': pos['side'],
                'market_value': float(pos['market_value']),
                'cost_basis': float(pos['cost_basis']),
                'unrealized_pnl': float(pos['unrealized_pl']),
                'unrealized_pnl_pct': float(pos['unrealized_plpc']),
                'current_price': float(pos['current_price']),
                'avg_entry_price': float(pos['avg_entry_price'])
            } for pos in positions]

        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            raise

    async def fetch_news(
        self,
        symbols: List[str],
        limit: int = 50
    ) -> List[Dict]:
        """Fetch news for symbols"""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            async with self.session.get(
                f"{self.DATA_URL}/v1beta1/news",
                params={
                    'symbols': ','.join(symbols),
                    'limit': limit
                }
            ) as response:
                data = await response.json()

            return [{
                'id': news['id'],
                'headline': news['headline'],
                'summary': news.get('summary', ''),
                'author': news.get('author', ''),
                'source': news['source'],
                'url': news['url'],
                'symbols': news.get('symbols', []),
                'created_at': news['created_at'],
                'updated_at': news.get('updated_at')
            } for news in data.get('news', [])]

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            raise


class YahooFinanceProvider(BaseDataProvider):
    """Free stock data provider using Yahoo Finance"""

    def __init__(self):
        super().__init__(Market.STOCKS)
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Initialize session"""
        try:
            self.session = aiohttp.ClientSession()
            self._connected = True
            logger.info("Yahoo Finance provider ready")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Yahoo Finance: {e}")
            return False

    async def disconnect(self) -> None:
        """Close session"""
        if self.session:
            await self.session.close()
            self._connected = False

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV from Yahoo Finance"""
        try:
            import yfinance as yf

            # Map timeframe
            interval_map = {
                TimeFrame.M1: '1m',
                TimeFrame.M5: '5m',
                TimeFrame.M15: '15m',
                TimeFrame.M30: '30m',
                TimeFrame.H1: '1h',
                TimeFrame.H4: '4h',  # Not supported, use 1h
                TimeFrame.D1: '1d',
                TimeFrame.W1: '1wk',
            }

            ticker = yf.Ticker(symbol)
            interval = interval_map.get(timeframe, '1d')

            # Calculate period
            if not start:
                start = datetime.now() - timedelta(days=365)
            if not end:
                end = datetime.now()

            df = ticker.history(
                start=start,
                end=end,
                interval=interval
            )

            if df.empty:
                return pd.DataFrame()

            # Rename columns to standard format
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]

            return self._validate_ohlcv(df)

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            raise
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current quote from Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'last': info.get('regularMarketPrice', 0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching Yahoo ticker for {symbol}: {e}")
            raise

    async def fetch_order_book(self, symbol: str, depth: int = 20) -> Dict[str, List]:
        """Yahoo Finance doesn't provide order book"""
        ticker = await self.fetch_ticker(symbol)
        return {
            'bids': [[ticker['bid'], 0]],
            'asks': [[ticker['ask'], 0]],
            'timestamp': ticker['timestamp']
        }

    async def get_available_symbols(self) -> List[str]:
        """Return common symbols (Yahoo doesn't have a list endpoint)"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD',
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI'
        ]

    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol info from Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'exchange': info.get('exchange', ''),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', 0),
                'min_quantity': 1,
                'tick_size': 0.01
            }
        except Exception as e:
            logger.error(f"Error fetching Yahoo symbol info for {symbol}: {e}")
            raise
