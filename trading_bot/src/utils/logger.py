"""
Logging Utilities - Structured logging for the trading bot
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: Use JSON formatting

    Returns:
        Root logger
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        root_logger.addHandler(file_handler)

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)

    return root_logger


class TradingLogger:
    """Specialized logger for trading operations"""

    def __init__(self, name: str = "trading"):
        self.logger = logging.getLogger(name)

    def trade(self, action: str, symbol: str, quantity: float, price: float, **kwargs):
        """Log a trade"""
        self.logger.info(
            f"TRADE: {action} {quantity} {symbol} @ {price}",
            extra={'trade': {'action': action, 'symbol': symbol, 'quantity': quantity, 'price': price, **kwargs}}
        )

    def signal(self, symbol: str, signal: int, confidence: float = None):
        """Log a trading signal"""
        signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(signal, 'UNKNOWN')
        msg = f"SIGNAL: {signal_name} {symbol}"
        if confidence:
            msg += f" (confidence: {confidence:.2%})"
        self.logger.info(msg)

    def risk_alert(self, message: str, level: str = "WARNING"):
        """Log a risk alert"""
        log_func = getattr(self.logger, level.lower(), self.logger.warning)
        log_func(f"RISK ALERT: {message}")

    def performance(self, metric: str, value: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            f"PERFORMANCE: {metric} = {value}",
            extra={'metric': metric, 'value': value, **kwargs}
        )
