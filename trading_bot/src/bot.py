"""
Trading Bot - Main orchestrator for the trading system
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import signal
import sys

from config import config, Market, TimeFrame
from src.data_pipeline import DataAggregator, data_aggregator
from src.features import FeatureEngineer
from src.models import EnsembleModel, model_registry
from src.risk import RiskManager
from src.execution import OrderManager, OrderSide, PaperTradingExecutor
from src.strategies import BaseStrategy, MLStrategy, EnsembleStrategy
from src.backtesting import BacktestEngine, print_backtest_report


logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.
    Coordinates data fetching, signal generation, risk management, and execution.
    """

    def __init__(
        self,
        strategy: BaseStrategy = None,
        paper_trading: bool = True
    ):
        self.strategy = strategy
        self.paper_trading = paper_trading

        # Components
        self.data_aggregator = data_aggregator
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager()

        # State
        self.is_running = False
        self.symbols: Dict[Market, List[str]] = {}
        self.last_signals: Dict[str, int] = {}
        self.positions: Dict[str, Dict] = {}

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.stop()

    async def initialize(
        self,
        markets: List[Market] = None,
        symbols: Dict[Market, List[str]] = None
    ) -> bool:
        """
        Initialize the trading bot.

        Args:
            markets: Markets to trade
            symbols: Symbols per market

        Returns:
            True if initialization successful
        """
        logger.info("Initializing trading bot...")

        # Initialize data providers
        markets = markets or config.trading.markets
        init_results = await self.data_aggregator.initialize(markets)

        if not any(init_results.values()):
            logger.error("Failed to initialize any data providers")
            return False

        # Set symbols
        if symbols:
            self.symbols = symbols
        else:
            self.symbols = {
                Market.CRYPTO: config.trading.crypto_symbols,
                Market.FOREX: config.trading.forex_symbols,
                Market.STOCKS: config.trading.stock_symbols
            }

        # Setup executors
        if self.paper_trading:
            for market in markets:
                executor = PaperTradingExecutor()
                self.order_manager.register_executor(market, executor)

        logger.info(f"Bot initialized with markets: {[m.value for m in init_results if init_results[m]]}")
        return True

    async def start(self, interval_seconds: int = 60):
        """
        Start the trading bot main loop.

        Args:
            interval_seconds: Seconds between trading cycles
        """
        self.is_running = True
        logger.info(f"Starting trading bot with {interval_seconds}s interval")

        while self.is_running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        logger.debug("Starting trading cycle")

        # Check risk limits
        can_trade, reason = self.risk_manager.check_risk_limits()
        if not can_trade:
            logger.warning(f"Trading halted: {reason}")
            return

        # Process each market
        for market, symbols in self.symbols.items():
            if market not in self.data_aggregator.get_connected_markets():
                continue

            for symbol in symbols:
                try:
                    await self._process_symbol(symbol, market)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

    async def _process_symbol(self, symbol: str, market: Market):
        """Process a single symbol"""
        # Fetch latest data
        data = await self.data_aggregator.fetch_ohlcv(
            symbol,
            config.trading.primary_timeframe,
            market=market,
            limit=200
        )

        if data.empty:
            logger.warning(f"No data for {symbol}")
            return

        # Generate signal
        if self.strategy:
            signal = self.strategy.generate_signal(data)
        else:
            signal = 0

        # Check if signal changed
        prev_signal = self.last_signals.get(symbol, 0)
        self.last_signals[symbol] = signal

        if signal == prev_signal:
            return  # No change

        # Get current price
        current_price = data['close'].iloc[-1]

        # Update paper trading executor with current price
        for executor in self.order_manager.executors.values():
            if hasattr(executor, 'set_price'):
                executor.set_price(symbol, current_price)

        # Execute signal
        if signal == 1 and prev_signal != 1:
            await self._open_long(symbol, market, current_price, data)
        elif signal == -1 and prev_signal != -1:
            await self._open_short(symbol, market, current_price, data)
        elif signal == 0 and prev_signal != 0:
            await self._close_position(symbol)

    async def _open_long(
        self,
        symbol: str,
        market: Market,
        price: float,
        data
    ):
        """Open a long position"""
        # Close existing position if any
        if symbol in self.positions:
            await self._close_position(symbol)

        # Calculate ATR for stop loss
        from src.features import TechnicalIndicators
        atr = TechnicalIndicators.atr(data['high'], data['low'], data['close']).iloc[-1]

        # Calculate position size
        stop_loss = self.risk_manager.calculate_stop_loss(price, 'long', atr=atr)
        size = self.risk_manager.calculate_position_size(symbol, price, stop_loss, market)

        # Check if we can open position
        position_value = size * price
        risk_amount = abs(price - stop_loss) * size

        can_open, reason = self.risk_manager.can_open_position(symbol, position_value, risk_amount)
        if not can_open:
            logger.warning(f"Cannot open position for {symbol}: {reason}")
            return

        # Create and submit order
        order = self.order_manager.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=size,
            market=market
        )

        success = await self.order_manager.submit_order(order)

        if success and order.filled_quantity > 0:
            take_profit = self.risk_manager.calculate_take_profit(price, stop_loss, 'long')

            # Register position with risk manager
            self.risk_manager.register_position(
                symbol=symbol,
                side='long',
                size=order.filled_quantity,
                entry_price=order.filled_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self.positions[symbol] = {
                'side': 'long',
                'size': order.filled_quantity,
                'entry_price': order.filled_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }

            logger.info(f"Opened LONG {symbol}: {order.filled_quantity} @ {order.filled_price}")

    async def _open_short(
        self,
        symbol: str,
        market: Market,
        price: float,
        data
    ):
        """Open a short position"""
        # Close existing position if any
        if symbol in self.positions:
            await self._close_position(symbol)

        # Calculate ATR for stop loss
        from src.features import TechnicalIndicators
        atr = TechnicalIndicators.atr(data['high'], data['low'], data['close']).iloc[-1]

        # Calculate position size
        stop_loss = self.risk_manager.calculate_stop_loss(price, 'short', atr=atr)
        size = self.risk_manager.calculate_position_size(symbol, price, stop_loss, market)

        # Check if we can open position
        position_value = size * price
        risk_amount = abs(price - stop_loss) * size

        can_open, reason = self.risk_manager.can_open_position(symbol, position_value, risk_amount)
        if not can_open:
            logger.warning(f"Cannot open position for {symbol}: {reason}")
            return

        # Create and submit order
        order = self.order_manager.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=size,
            market=market
        )

        success = await self.order_manager.submit_order(order)

        if success and order.filled_quantity > 0:
            take_profit = self.risk_manager.calculate_take_profit(price, stop_loss, 'short')

            # Register position with risk manager
            self.risk_manager.register_position(
                symbol=symbol,
                side='short',
                size=order.filled_quantity,
                entry_price=order.filled_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self.positions[symbol] = {
                'side': 'short',
                'size': order.filled_quantity,
                'entry_price': order.filled_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }

            logger.info(f"Opened SHORT {symbol}: {order.filled_quantity} @ {order.filled_price}")

    async def _close_position(self, symbol: str):
        """Close an existing position"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Create close order
        side = OrderSide.SELL if position['side'] == 'long' else OrderSide.BUY

        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            quantity=position['size']
        )

        success = await self.order_manager.submit_order(order)

        if success:
            # Update risk manager
            self.risk_manager.close_position(symbol)
            del self.positions[symbol]

            logger.info(f"Closed position {symbol}: {order.filled_quantity} @ {order.filled_price}")

    async def backtest(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        market: Market = None
    ):
        """
        Run backtest for a symbol.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            market: Market type
        """
        from datetime import datetime

        start = datetime.strptime(start_date or config.backtest.start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date or config.backtest.end_date, '%Y-%m-%d')

        logger.info(f"Running backtest for {symbol} from {start} to {end}")

        # Fetch historical data
        data = await self.data_aggregator.fetch_ohlcv(
            symbol,
            config.trading.primary_timeframe,
            market=market,
            start=start,
            end=end,
            limit=10000
        )

        if data.empty:
            logger.error(f"No data available for {symbol}")
            return

        # Run backtest
        engine = BacktestEngine(
            initial_capital=config.backtest.initial_capital,
            commission=config.backtest.commission_rate,
            slippage=config.backtest.slippage_bps / 10000
        )

        if self.strategy:
            result = engine.run(data, self.strategy.generate_signal, symbol)
            print_backtest_report(result)
            return result

        logger.warning("No strategy set for backtest")

    async def train_model(
        self,
        symbols: List[str] = None,
        market: Market = None
    ):
        """
        Train ML model on historical data.

        Args:
            symbols: Symbols to train on
            market: Market type
        """
        from datetime import datetime, timedelta

        symbols = symbols or self.symbols.get(market or Market.STOCKS, [])[:3]

        logger.info(f"Training model on {symbols}")

        all_features = []

        for symbol in symbols:
            # Fetch historical data
            data = await self.data_aggregator.fetch_ohlcv(
                symbol,
                config.trading.primary_timeframe,
                market=market,
                limit=5000
            )

            if data.empty:
                continue

            # Generate features
            features = self.feature_engineer.generate_features(data)
            all_features.append(features)

        if not all_features:
            logger.error("No data available for training")
            return

        import pandas as pd
        import numpy as np

        # Combine features
        combined = pd.concat(all_features, ignore_index=True)

        # Split data
        train_size = int(len(combined) * config.model.train_test_split)
        val_size = int(len(combined) * config.model.validation_split)

        train_data = combined.iloc[:train_size]
        val_data = combined.iloc[train_size:train_size + val_size]
        test_data = combined.iloc[train_size + val_size:]

        # Get feature matrix
        X_train, y_train = self.feature_engineer.get_feature_matrix(train_data)
        X_val, y_val = self.feature_engineer.get_feature_matrix(val_data)
        X_test, y_test = self.feature_engineer.get_feature_matrix(test_data)

        # Convert target to classification
        y_train = np.nan_to_num(train_data['target_class'].values.astype(int))
        y_val = np.nan_to_num(val_data['target_class'].values.astype(int))
        y_test = np.nan_to_num(test_data['target_class'].values.astype(int))

        # Build and train ensemble model
        model = EnsembleModel()
        model.build(
            input_shape=(X_train.shape[1],),
            include_lstm=False,  # Disable for faster training
            include_transformer=False,
            num_classes=3
        )

        results = model.train(X_train, y_train, X_val, y_val)

        # Evaluate
        eval_results = model.evaluate(X_test, y_test)
        logger.info(f"Model evaluation: {eval_results}")

        # Save model
        model.save()

        # Update strategy
        if isinstance(self.strategy, MLStrategy):
            self.strategy.set_model(model)

        model_registry.register(model)

        return model

    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'paper_trading': self.paper_trading,
            'connected_markets': [m.value for m in self.data_aggregator.get_connected_markets()],
            'active_positions': len(self.positions),
            'positions': self.positions,
            'risk_metrics': self.risk_manager.get_risk_metrics().__dict__,
            'active_orders': len(self.order_manager.get_active_orders())
        }

    async def shutdown(self):
        """Cleanup and shutdown"""
        logger.info("Shutting down trading bot...")

        # Close all positions
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol)

        # Disconnect data providers
        await self.data_aggregator.shutdown()

        logger.info("Trading bot shutdown complete")


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create strategy
    from src.strategies import MomentumStrategy
    strategy = MomentumStrategy()

    # Create bot
    bot = TradingBot(strategy=strategy, paper_trading=True)

    # Initialize
    success = await bot.initialize(
        markets=[Market.CRYPTO],
        symbols={Market.CRYPTO: ['BTC/USDT', 'ETH/USDT']}
    )

    if not success:
        logger.error("Failed to initialize bot")
        return

    # Run backtest
    result = await bot.backtest('BTC/USDT', '2023-01-01', '2024-01-01', Market.CRYPTO)

    # Or start live/paper trading
    # await bot.start(interval_seconds=60)

    await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
