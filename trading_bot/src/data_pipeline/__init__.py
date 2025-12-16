"""
Data Pipeline Module
Provides unified data access for Crypto, Forex, and Stock markets.
"""

from .base_provider import BaseDataProvider
from .crypto_provider import CryptoDataProvider, MultiExchangeCryptoProvider
from .forex_provider import OANDADataProvider
from .stock_provider import AlpacaDataProvider, YahooFinanceProvider
from .aggregator import DataAggregator, data_aggregator

__all__ = [
    'BaseDataProvider',
    'CryptoDataProvider',
    'MultiExchangeCryptoProvider',
    'OANDADataProvider',
    'AlpacaDataProvider',
    'YahooFinanceProvider',
    'DataAggregator',
    'data_aggregator'
]
