#!/usr/bin/env python3
"""
Institutional AI Trading Bot - Main Entry Point

Usage:
    python main.py --mode paper --market crypto
    python main.py --mode backtest --symbol BTC/USDT --start 2023-01-01
    python main.py --mode train --market stocks
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
from src.bot import TradingBot
from src.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    MLStrategy,
    EnsembleStrategy
)
from config import Market, config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Institutional AI Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest', 'train'],
        default='paper',
        help='Trading mode (default: paper)'
    )

    parser.add_argument(
        '--market',
        choices=['crypto', 'forex', 'stocks', 'all'],
        default='crypto',
        help='Market to trade (default: crypto)'
    )

    parser.add_argument(
        '--strategy',
        choices=['momentum', 'mean_reversion', 'trend', 'ml', 'ensemble'],
        default='momentum',
        help='Trading strategy (default: momentum)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Symbol for backtest (e.g., BTC/USDT, AAPL)'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2024-01-01',
        help='Backtest end date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Trading interval in seconds (default: 60)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path (optional)'
    )

    return parser.parse_args()


def get_markets(market_arg: str):
    """Convert market argument to Market enum list"""
    if market_arg == 'all':
        return [Market.CRYPTO, Market.FOREX, Market.STOCKS]
    market_map = {
        'crypto': Market.CRYPTO,
        'forex': Market.FOREX,
        'stocks': Market.STOCKS
    }
    return [market_map[market_arg]]


def get_strategy(strategy_arg: str):
    """Create strategy instance from argument"""
    strategies = {
        'momentum': MomentumStrategy(),
        'mean_reversion': MeanReversionStrategy(),
        'trend': TrendFollowingStrategy(),
        'ml': MLStrategy(),
        'ensemble': EnsembleStrategy([
            MomentumStrategy(),
            TrendFollowingStrategy()
        ])
    }
    return strategies.get(strategy_arg, MomentumStrategy())


def get_symbols(market: Market):
    """Get default symbols for market"""
    return {
        Market.CRYPTO: config.trading.crypto_symbols,
        Market.FOREX: config.trading.forex_symbols,
        Market.STOCKS: config.trading.stock_symbols
    }.get(market, [])


async def run_paper_trading(bot: TradingBot, args):
    """Run paper trading mode"""
    markets = get_markets(args.market)

    symbols = {}
    for market in markets:
        symbols[market] = get_symbols(market)

    success = await bot.initialize(markets=markets, symbols=symbols)
    if not success:
        print("Failed to initialize bot")
        return

    print(f"\n{'='*60}")
    print("PAPER TRADING MODE")
    print(f"{'='*60}")
    print(f"Markets: {[m.value for m in markets]}")
    print(f"Strategy: {bot.strategy.name}")
    print(f"Interval: {args.interval}s")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop\n")

    try:
        await bot.start(interval_seconds=args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown()


async def run_backtest(bot: TradingBot, args):
    """Run backtest mode"""
    market = get_markets(args.market)[0]

    if not args.symbol:
        # Use first symbol from market
        args.symbol = get_symbols(market)[0]

    success = await bot.initialize(markets=[market])
    if not success:
        print("Failed to initialize bot")
        return

    print(f"\n{'='*60}")
    print("BACKTEST MODE")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Strategy: {bot.strategy.name}")
    print(f"{'='*60}\n")

    result = await bot.backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        market=market
    )

    await bot.shutdown()
    return result


async def run_training(bot: TradingBot, args):
    """Run model training mode"""
    market = get_markets(args.market)[0]

    success = await bot.initialize(markets=[market])
    if not success:
        print("Failed to initialize bot")
        return

    symbols = get_symbols(market)[:5]  # Use first 5 symbols

    print(f"\n{'='*60}")
    print("TRAINING MODE")
    print(f"{'='*60}")
    print(f"Market: {market.value}")
    print(f"Symbols: {symbols}")
    print(f"{'='*60}\n")

    model = await bot.train_model(symbols=symbols, market=market)

    await bot.shutdown()
    return model


async def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    log_file = args.log_file or f"logs/trading_{args.mode}.log"
    setup_logging(level=args.log_level, log_file=log_file)

    # Create strategy and bot
    strategy = get_strategy(args.strategy)
    paper_trading = args.mode != 'live'

    bot = TradingBot(strategy=strategy, paper_trading=paper_trading)

    # Run selected mode
    if args.mode in ['paper', 'live']:
        await run_paper_trading(bot, args)
    elif args.mode == 'backtest':
        await run_backtest(bot, args)
    elif args.mode == 'train':
        await run_training(bot, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
