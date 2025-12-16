"""
Backtesting Engine - Comprehensive strategy backtesting system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from config import config, Market


logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_period: int = 0
    status: str = 'open'  # 'open', 'closed', 'stopped'

    def close(self, exit_price: float, exit_time: datetime, commission: float = 0):
        """Close the trade"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.commission += commission

        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.commission
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.commission
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price

        self.status = 'closed'


@dataclass
class Position:
    """Current position state"""
    symbol: str
    side: str  # 'long', 'short', 'flat'
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Backtesting results"""
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0

    # Time series
    equity_curve: pd.Series = None
    drawdown_curve: pd.Series = None
    returns: pd.Series = None

    # Trade log
    trades: List[Trade] = field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtesting engine with realistic execution simulation.
    """

    def __init__(
        self,
        initial_capital: float = None,
        commission: float = None,
        slippage: float = None
    ):
        self.initial_capital = initial_capital or config.backtest.initial_capital
        self.commission = commission or config.backtest.commission_rate
        self.slippage = slippage or config.backtest.slippage_bps / 10000

        self.reset()

    def reset(self):
        """Reset engine state"""
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Dict] = []
        self.current_time: Optional[datetime] = None

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        symbol: str = "SYMBOL",
        **strategy_kwargs
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            data: DataFrame with OHLCV data (must have datetime index)
            strategy: Strategy function that returns signals
            symbol: Trading symbol
            **strategy_kwargs: Additional arguments for strategy

        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        self.positions[symbol] = Position(symbol=symbol, side='flat')

        logger.info(f"Starting backtest for {symbol} from {data.index[0]} to {data.index[-1]}")

        for i in range(len(data)):
            row = data.iloc[i]
            self.current_time = data.index[i]

            # Get historical data up to current point
            history = data.iloc[:i+1]

            # Get signal from strategy
            signal = strategy(history, **strategy_kwargs)

            # Execute signal
            self._execute_signal(signal, row, symbol)

            # Update equity
            self._update_equity(row, symbol)

        # Close any open positions at end
        self._close_all_positions(data.iloc[-1])

        # Calculate results
        return self._calculate_results()

    def _execute_signal(
        self,
        signal: int,
        row: pd.Series,
        symbol: str
    ):
        """
        Execute trading signal

        Args:
            signal: 1 (buy), -1 (sell), 0 (hold)
            row: Current OHLCV row
            symbol: Trading symbol
        """
        position = self.positions[symbol]
        price = row['close']

        # Apply slippage
        if signal == 1:
            exec_price = price * (1 + self.slippage)
        elif signal == -1:
            exec_price = price * (1 - self.slippage)
        else:
            exec_price = price

        # Calculate position size (fixed fraction for simplicity)
        position_value = self.capital * config.risk.max_position_size

        if signal == 1 and position.side != 'long':
            # Close short if exists
            if position.side == 'short':
                self._close_position(symbol, exec_price)

            # Open long
            quantity = position_value / exec_price
            commission = position_value * self.commission
            self.capital -= (position_value + commission)

            position.side = 'long'
            position.quantity = quantity
            position.avg_price = exec_price

            self.trades.append(Trade(
                entry_time=self.current_time,
                exit_time=None,
                symbol=symbol,
                side='long',
                entry_price=exec_price,
                exit_price=None,
                quantity=quantity,
                commission=commission
            ))

        elif signal == -1 and position.side != 'short':
            # Close long if exists
            if position.side == 'long':
                self._close_position(symbol, exec_price)

            # Open short
            quantity = position_value / exec_price
            commission = position_value * self.commission
            self.capital -= commission  # Only commission for short

            position.side = 'short'
            position.quantity = quantity
            position.avg_price = exec_price

            self.trades.append(Trade(
                entry_time=self.current_time,
                exit_time=None,
                symbol=symbol,
                side='short',
                entry_price=exec_price,
                exit_price=None,
                quantity=quantity,
                commission=commission
            ))

    def _close_position(self, symbol: str, price: float):
        """Close current position"""
        position = self.positions[symbol]

        if position.side == 'flat':
            return

        # Find open trade
        open_trade = next(
            (t for t in reversed(self.trades) if t.symbol == symbol and t.status == 'open'),
            None
        )

        if open_trade:
            commission = position.quantity * price * self.commission
            open_trade.close(price, self.current_time, commission)

            # Update capital
            if position.side == 'long':
                self.capital += position.quantity * price - commission
            else:  # short
                profit = (open_trade.entry_price - price) * position.quantity
                self.capital += open_trade.entry_price * position.quantity + profit - commission

            position.realized_pnl += open_trade.pnl

        # Reset position
        position.side = 'flat'
        position.quantity = 0
        position.avg_price = 0

    def _close_all_positions(self, row: pd.Series):
        """Close all open positions"""
        price = row['close']
        for symbol in self.positions:
            self._close_position(symbol, price)

    def _update_equity(self, row: pd.Series, symbol: str):
        """Update equity curve"""
        position = self.positions[symbol]
        price = row['close']

        # Calculate unrealized PnL
        if position.side == 'long':
            unrealized = (price - position.avg_price) * position.quantity
        elif position.side == 'short':
            unrealized = (position.avg_price - price) * position.quantity
        else:
            unrealized = 0

        position.unrealized_pnl = unrealized

        # Total equity
        total_equity = self.capital + unrealized

        self.equity_history.append({
            'timestamp': self.current_time,
            'equity': total_equity,
            'capital': self.capital,
            'unrealized_pnl': unrealized,
            'position': position.side
        })

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest metrics"""
        result = BacktestResult()

        # Convert equity history to DataFrame
        equity_df = pd.DataFrame(self.equity_history)
        if equity_df.empty:
            return result

        equity_df.set_index('timestamp', inplace=True)
        result.equity_curve = equity_df['equity']

        # Calculate returns
        result.returns = result.equity_curve.pct_change().dropna()

        # Total return
        result.total_return = (result.equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days > 0:
            result.annual_return = (1 + result.total_return) ** (365 / days) - 1

        # Sharpe ratio (assuming 0 risk-free rate)
        if len(result.returns) > 0 and result.returns.std() > 0:
            result.sharpe_ratio = result.returns.mean() / result.returns.std() * np.sqrt(252)

        # Sortino ratio
        downside_returns = result.returns[result.returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            result.sortino_ratio = result.returns.mean() / downside_returns.std() * np.sqrt(252)

        # Drawdown
        rolling_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - rolling_max) / rolling_max
        result.drawdown_curve = drawdown
        result.max_drawdown = drawdown.min()

        # Calmar ratio
        if result.max_drawdown != 0:
            result.calmar_ratio = result.annual_return / abs(result.max_drawdown)

        # Trade statistics
        closed_trades = [t for t in self.trades if t.status == 'closed']
        result.trades = closed_trades
        result.total_trades = len(closed_trades)

        if closed_trades:
            pnls = [t.pnl for t in closed_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = len(wins) / len(closed_trades) if closed_trades else 0

            result.avg_win = np.mean(wins) if wins else 0
            result.avg_loss = np.mean(losses) if losses else 0
            result.largest_win = max(pnls) if pnls else 0
            result.largest_loss = min(pnls) if pnls else 0

            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            # Average holding period
            holding_periods = [(t.exit_time - t.entry_time).days for t in closed_trades if t.exit_time]
            result.avg_holding_period = np.mean(holding_periods) if holding_periods else 0

        return result


class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust strategy validation
    """

    def __init__(
        self,
        engine: BacktestEngine,
        in_sample_size: float = 0.7,
        out_sample_size: float = 0.3,
        n_splits: int = 5
    ):
        self.engine = engine
        self.in_sample_size = in_sample_size
        self.out_sample_size = out_sample_size
        self.n_splits = n_splits

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_class,
        param_grid: Dict[str, List],
        symbol: str = "SYMBOL",
        metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization

        Args:
            data: Full historical data
            strategy_class: Strategy class to optimize
            param_grid: Parameter grid for optimization
            symbol: Trading symbol
            metric: Metric to optimize

        Returns:
            Optimization results
        """
        from itertools import product

        results = []
        total_len = len(data)
        split_size = total_len // self.n_splits

        for split in range(self.n_splits - 1):
            # Define in-sample and out-of-sample periods
            is_start = split * split_size
            is_end = is_start + int(split_size * self.in_sample_size)
            os_start = is_end
            os_end = (split + 1) * split_size

            in_sample = data.iloc[is_start:is_end]
            out_sample = data.iloc[os_start:os_end]

            # Grid search on in-sample
            best_params = None
            best_score = -np.inf

            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())

            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                strategy = strategy_class(**param_dict)

                result = self.engine.run(in_sample, strategy.generate_signal, symbol)
                score = getattr(result, metric, 0)

                if score > best_score:
                    best_score = score
                    best_params = param_dict

            # Test on out-of-sample
            if best_params:
                strategy = strategy_class(**best_params)
                os_result = self.engine.run(out_sample, strategy.generate_signal, symbol)

                results.append({
                    'split': split,
                    'in_sample_score': best_score,
                    'out_sample_score': getattr(os_result, metric, 0),
                    'params': best_params,
                    'result': os_result
                })

        return {
            'splits': results,
            'avg_is_score': np.mean([r['in_sample_score'] for r in results]),
            'avg_os_score': np.mean([r['out_sample_score'] for r in results]),
            'best_params': max(results, key=lambda x: x['out_sample_score'])['params']
        }


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\n{'PERFORMANCE METRICS':^60}")
    print("-"*60)
    print(f"Total Return:        {result.total_return*100:>10.2f}%")
    print(f"Annual Return:       {result.annual_return*100:>10.2f}%")
    print(f"Sharpe Ratio:        {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:       {result.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:        {result.max_drawdown*100:>10.2f}%")
    print(f"Calmar Ratio:        {result.calmar_ratio:>10.2f}")

    print(f"\n{'TRADE STATISTICS':^60}")
    print("-"*60)
    print(f"Total Trades:        {result.total_trades:>10}")
    print(f"Winning Trades:      {result.winning_trades:>10}")
    print(f"Losing Trades:       {result.losing_trades:>10}")
    print(f"Win Rate:            {result.win_rate*100:>10.2f}%")
    print(f"Profit Factor:       {result.profit_factor:>10.2f}")
    print(f"Avg Win:             ${result.avg_win:>9.2f}")
    print(f"Avg Loss:            ${result.avg_loss:>9.2f}")
    print(f"Largest Win:         ${result.largest_win:>9.2f}")
    print(f"Largest Loss:        ${result.largest_loss:>9.2f}")
    print(f"Avg Holding Period:  {result.avg_holding_period:>10.1f} days")

    print("="*60 + "\n")
