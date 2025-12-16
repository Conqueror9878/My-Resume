"""
Forex Data Provider - OANDA and Interactive Brokers integration
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


OANDA_TIMEFRAME_MAP = {
    TimeFrame.M1: 'M1',
    TimeFrame.M5: 'M5',
    TimeFrame.M15: 'M15',
    TimeFrame.M30: 'M30',
    TimeFrame.H1: 'H1',
    TimeFrame.H4: 'H4',
    TimeFrame.D1: 'D',
    TimeFrame.W1: 'W',
}


class OANDADataProvider(BaseDataProvider):
    """Forex data provider using OANDA API"""

    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    def __init__(self, practice: bool = True):
        super().__init__(Market.FOREX)
        self.base_url = self.PRACTICE_URL if practice else self.LIVE_URL
        self.api_key = config.api.oanda_api_key
        self.account_id = config.api.oanda_account_id
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Connect to OANDA API"""
        if not self.api_key or not self.account_id:
            logger.error("OANDA API key or account ID not configured")
            return False

        try:
            self.session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            )

            # Test connection
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}"
            ) as response:
                if response.status == 200:
                    self._connected = True
                    logger.info("Connected to OANDA")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"OANDA connection failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from OANDA"""
        if self.session:
            await self.session.close()
            self._connected = False
            logger.info("Disconnected from OANDA")

    def _format_symbol(self, symbol: str) -> str:
        """Convert standard format to OANDA format (EUR/USD -> EUR_USD)"""
        return symbol.replace('/', '_')

    def _parse_symbol(self, symbol: str) -> str:
        """Convert OANDA format to standard format (EUR_USD -> EUR/USD)"""
        return symbol.replace('_', '/')

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data from OANDA"""
        if not self._connected:
            raise ConnectionError("Not connected to OANDA")

        oanda_symbol = self._format_symbol(symbol)
        granularity = OANDA_TIMEFRAME_MAP.get(timeframe, 'H1')

        params = {
            'granularity': granularity,
            'count': min(limit, 5000)
        }

        if start:
            params['from'] = start.strftime('%Y-%m-%dT%H:%M:%SZ')
        if end:
            params['to'] = end.strftime('%Y-%m-%dT%H:%M:%SZ')

        try:
            async with self.session.get(
                f"{self.base_url}/v3/instruments/{oanda_symbol}/candles",
                params=params
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"OANDA API error: {error}")

                data = await response.json()
                candles = data.get('candles', [])

                records = []
                for candle in candles:
                    if candle.get('complete', True):
                        mid = candle.get('mid', {})
                        records.append({
                            'timestamp': pd.to_datetime(candle['time']),
                            'open': float(mid.get('o', 0)),
                            'high': float(mid.get('h', 0)),
                            'low': float(mid.get('l', 0)),
                            'close': float(mid.get('c', 0)),
                            'volume': int(candle.get('volume', 0))
                        })

                df = pd.DataFrame(records)
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    return self._validate_ohlcv(df)
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current price"""
        if not self._connected:
            raise ConnectionError("Not connected to OANDA")

        oanda_symbol = self._format_symbol(symbol)

        try:
            async with self.session.get(
                f"{self.base_url}/v3/instruments/{oanda_symbol}/candles",
                params={'count': 1, 'granularity': 'S5'}
            ) as response:
                data = await response.json()

            # Get pricing
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/pricing",
                params={'instruments': oanda_symbol}
            ) as response:
                pricing = await response.json()

            price_data = pricing.get('prices', [{}])[0]

            return {
                'bid': float(price_data.get('bids', [{}])[0].get('price', 0)),
                'ask': float(price_data.get('asks', [{}])[0].get('price', 0)),
                'last': float(price_data.get('closeoutBid', 0)),
                'spread': None,  # Calculated from bid/ask
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
        """Fetch order book (position book for forex)"""
        if not self._connected:
            raise ConnectionError("Not connected to OANDA")

        oanda_symbol = self._format_symbol(symbol)

        try:
            async with self.session.get(
                f"{self.base_url}/v3/instruments/{oanda_symbol}/orderBook"
            ) as response:
                data = await response.json()

            buckets = data.get('orderBook', {}).get('buckets', [])

            bids = []
            asks = []
            for bucket in buckets:
                price = float(bucket['price'])
                long_pct = float(bucket.get('longCountPercent', 0))
                short_pct = float(bucket.get('shortCountPercent', 0))
                if long_pct > 0:
                    bids.append([price, long_pct])
                if short_pct > 0:
                    asks.append([price, short_pct])

            return {
                'bids': sorted(bids, key=lambda x: x[0], reverse=True)[:depth],
                'asks': sorted(asks, key=lambda x: x[0])[:depth],
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise

    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable forex pairs"""
        if not self._connected:
            raise ConnectionError("Not connected to OANDA")

        try:
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
            ) as response:
                data = await response.json()

            instruments = data.get('instruments', [])
            return [
                self._parse_symbol(inst['name'])
                for inst in instruments
                if inst.get('type') == 'CURRENCY'
            ]

        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            raise

    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol trading info"""
        if not self._connected:
            raise ConnectionError("Not connected to OANDA")

        oanda_symbol = self._format_symbol(symbol)

        try:
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/instruments",
                params={'instruments': oanda_symbol}
            ) as response:
                data = await response.json()

            inst = data.get('instruments', [{}])[0]

            return {
                'symbol': symbol,
                'type': inst.get('type'),
                'display_name': inst.get('displayName'),
                'pip_location': inst.get('pipLocation'),
                'min_quantity': float(inst.get('minimumTradeSize', 1)),
                'max_quantity': float(inst.get('maximumOrderUnits', 100000000)),
                'margin_rate': float(inst.get('marginRate', 0.02)),
                'tick_size': 10 ** inst.get('pipLocation', -4),
            }

        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            raise

    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not self._connected:
            raise ConnectionError("Not connected to OANDA")

        try:
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}"
            ) as response:
                data = await response.json()

            account = data.get('account', {})
            return {
                'balance': float(account.get('balance', 0)),
                'nav': float(account.get('NAV', 0)),
                'unrealized_pnl': float(account.get('unrealizedPL', 0)),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'position_count': int(account.get('openPositionCount', 0)),
                'currency': account.get('currency', 'USD')
            }

        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise

    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self._connected:
            raise ConnectionError("Not connected to OANDA")

        try:
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
            ) as response:
                data = await response.json()

            positions = []
            for pos in data.get('positions', []):
                long_units = int(pos.get('long', {}).get('units', 0))
                short_units = int(pos.get('short', {}).get('units', 0))

                positions.append({
                    'symbol': self._parse_symbol(pos['instrument']),
                    'long_units': long_units,
                    'short_units': short_units,
                    'net_units': long_units + short_units,
                    'unrealized_pnl': float(pos.get('unrealizedPL', 0)),
                    'margin_used': float(pos.get('marginUsed', 0))
                })

            return positions

        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            raise
