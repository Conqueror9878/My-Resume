"""
Order Manager - Order routing and execution across markets
"""

import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

from config import config, Market, OrderType


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]  # None for market orders
    stop_price: Optional[float]  # For stop orders
    market: Market
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


@dataclass
class Fill:
    """Represents an order fill"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime


class OrderManager:
    """
    Manages order lifecycle and execution across multiple markets.
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_callbacks: List[Callable] = []
        self.executors: Dict[Market, 'BaseExecutor'] = {}

    def register_executor(self, market: Market, executor: 'BaseExecutor'):
        """Register an executor for a market"""
        self.executors[market] = executor
        logger.info(f"Registered executor for {market.value}")

    def add_callback(self, callback: Callable):
        """Add order update callback"""
        self.order_callbacks.append(callback)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        stop_price: float = None,
        market: Market = None
    ) -> Order:
        """
        Create a new order

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            market: Market type (auto-detected if not specified)

        Returns:
            Created Order object
        """
        # Auto-detect market if not specified
        if market is None:
            market = self._detect_market(symbol)

        order_id = str(uuid.uuid4())[:8]

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            market=market
        )

        self.orders[order_id] = order
        logger.info(f"Created order {order_id}: {side.value} {quantity} {symbol} @ {price or 'MARKET'}")

        return order

    async def submit_order(self, order: Order) -> bool:
        """
        Submit order to exchange

        Returns:
            True if submission successful
        """
        if order.market not in self.executors:
            order.status = OrderStatus.REJECTED
            order.error_message = f"No executor for market: {order.market}"
            logger.error(order.error_message)
            return False

        executor = self.executors[order.market]

        try:
            success = await executor.submit_order(order)

            if success:
                order.status = OrderStatus.SUBMITTED
                order.updated_at = datetime.now()
                logger.info(f"Order {order.id} submitted successfully")
            else:
                order.status = OrderStatus.REJECTED
                logger.error(f"Order {order.id} submission failed")

            await self._notify_callbacks(order)
            return success

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Order {order.id} submission error: {e}")
            await self._notify_callbacks(order)
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if not order.is_active:
            logger.warning(f"Order {order_id} is not active")
            return False

        executor = self.executors.get(order.market)
        if not executor:
            return False

        try:
            success = await executor.cancel_order(order)

            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                logger.info(f"Order {order_id} cancelled")

            await self._notify_callbacks(order)
            return success

        except Exception as e:
            logger.error(f"Cancel order {order_id} error: {e}")
            return False

    async def update_order_status(self, order_id: str) -> Optional[Order]:
        """Update order status from exchange"""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]
        executor = self.executors.get(order.market)

        if not executor:
            return order

        try:
            await executor.update_order_status(order)
            await self._notify_callbacks(order)
            return order
        except Exception as e:
            logger.error(f"Update order status error: {e}")
            return order

    def record_fill(
        self,
        order: Order,
        quantity: float,
        price: float,
        commission: float
    ):
        """Record an order fill"""
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            timestamp=datetime.now()
        )

        self.fills.append(fill)

        # Update order
        order.filled_quantity += quantity
        order.commission += commission

        # Calculate average fill price
        total_value = order.filled_price * (order.filled_quantity - quantity) + price * quantity
        order.filled_price = total_value / order.filled_quantity

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
        else:
            order.status = OrderStatus.PARTIAL

        order.updated_at = datetime.now()
        logger.info(f"Fill recorded for order {order.id}: {quantity} @ {price}")

    async def _notify_callbacks(self, order: Order):
        """Notify all callbacks of order update"""
        for callback in self.order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _detect_market(self, symbol: str) -> Market:
        """Detect market from symbol format"""
        if '/' in symbol:
            base, quote = symbol.split('/')
            forex_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'}
            if base in forex_currencies and quote in forex_currencies:
                return Market.FOREX
            return Market.CRYPTO
        return Market.STOCKS

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol"""
        orders = [o for o in self.orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_fills(self, symbol: str = None) -> List[Fill]:
        """Get all fills, optionally filtered by symbol"""
        if symbol:
            return [f for f in self.fills if f.symbol == symbol]
        return self.fills.copy()


class BaseExecutor:
    """Base class for market executors"""

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading

    async def submit_order(self, order: Order) -> bool:
        """Submit order to exchange"""
        raise NotImplementedError

    async def cancel_order(self, order: Order) -> bool:
        """Cancel order on exchange"""
        raise NotImplementedError

    async def update_order_status(self, order: Order) -> None:
        """Update order status from exchange"""
        raise NotImplementedError


class PaperTradingExecutor(BaseExecutor):
    """Paper trading executor for testing"""

    def __init__(self, slippage: float = 0.001):
        super().__init__(paper_trading=True)
        self.slippage = slippage
        self.current_prices: Dict[str, float] = {}

    def set_price(self, symbol: str, price: float):
        """Set current price for paper trading"""
        self.current_prices[symbol] = price

    async def submit_order(self, order: Order) -> bool:
        """Simulate order submission"""
        await asyncio.sleep(0.1)  # Simulate network latency

        price = self.current_prices.get(order.symbol)
        if not price:
            order.error_message = "No price available"
            return False

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage)
        else:
            fill_price = price * (1 - self.slippage)

        # For market orders, fill immediately
        if order.order_type == OrderType.MARKET:
            order.broker_order_id = f"PAPER-{order.id}"
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.commission = order.quantity * fill_price * 0.001
            order.filled_at = datetime.now()
            return True

        # For limit orders, check if price is favorable
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and price <= order.price:
                order.broker_order_id = f"PAPER-{order.id}"
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_price = order.price
                order.filled_at = datetime.now()
            elif order.side == OrderSide.SELL and price >= order.price:
                order.broker_order_id = f"PAPER-{order.id}"
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_price = order.price
                order.filled_at = datetime.now()

        return True

    async def cancel_order(self, order: Order) -> bool:
        """Simulate order cancellation"""
        await asyncio.sleep(0.05)
        return True

    async def update_order_status(self, order: Order) -> None:
        """Update order status (no-op for paper trading)"""
        pass


class CryptoExecutor(BaseExecutor):
    """Executor for cryptocurrency exchanges"""

    def __init__(self, exchange, paper_trading: bool = True):
        super().__init__(paper_trading)
        self.exchange = exchange

    async def submit_order(self, order: Order) -> bool:
        """Submit order to crypto exchange"""
        if self.paper_trading:
            return await self._paper_submit(order)

        try:
            if order.order_type == OrderType.MARKET:
                result = await self.exchange.create_market_order(
                    order.symbol,
                    order.side.value,
                    order.quantity
                )
            elif order.order_type == OrderType.LIMIT:
                result = await self.exchange.create_limit_order(
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.price
                )
            else:
                return False

            order.broker_order_id = result['id']
            return True

        except Exception as e:
            order.error_message = str(e)
            logger.error(f"Crypto order error: {e}")
            return False

    async def cancel_order(self, order: Order) -> bool:
        """Cancel order on exchange"""
        if self.paper_trading:
            return True

        try:
            await self.exchange.cancel_order(order.broker_order_id, order.symbol)
            return True
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False

    async def update_order_status(self, order: Order) -> None:
        """Update order status from exchange"""
        if self.paper_trading:
            return

        try:
            result = await self.exchange.fetch_order(order.broker_order_id, order.symbol)

            status_map = {
                'open': OrderStatus.SUBMITTED,
                'closed': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED
            }

            order.status = status_map.get(result['status'], order.status)
            order.filled_quantity = result.get('filled', 0)
            order.filled_price = result.get('average', 0)

        except Exception as e:
            logger.error(f"Update order status error: {e}")

    async def _paper_submit(self, order: Order) -> bool:
        """Paper trading submission"""
        try:
            ticker = await self.exchange.fetch_ticker(order.symbol)
            price = ticker['last']

            if order.side == OrderSide.BUY:
                fill_price = ticker['ask'] or price
            else:
                fill_price = ticker['bid'] or price

            order.broker_order_id = f"PAPER-{order.id}"
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.filled_at = datetime.now()

            return True

        except Exception as e:
            order.error_message = str(e)
            return False
