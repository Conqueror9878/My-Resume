"""
Risk Management Module - Position sizing, stop-loss, and portfolio risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from config import config, Market


logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    total_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    portfolio_var: float = 0.0
    portfolio_beta: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    positions_at_risk: int = 0
    margin_used: float = 0.0
    margin_available: float = 0.0


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_amount: float  # Amount at risk (to stop loss)
    risk_pct: float  # Risk as % of portfolio
    var_95: float  # 95% VaR for position
    expected_shortfall: float


class RiskManager:
    """
    Institutional-grade risk management system.
    Handles position sizing, stop-loss calculation, and portfolio risk.
    """

    def __init__(self, portfolio_value: float = None):
        self.portfolio_value = portfolio_value or config.backtest.initial_capital
        self.positions: Dict[str, PositionRisk] = {}
        self.daily_pnl_history: List[float] = []
        self.equity_high = self.portfolio_value
        self.risk_config = config.risk

        # Risk state
        self.is_trading_halted = False
        self.halt_reason: Optional[str] = None

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        market: Market = Market.STOCKS,
        volatility: float = None
    ) -> float:
        """
        Calculate optimal position size using multiple methods.

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            market: Market type for leverage limits
            volatility: Asset volatility (optional)

        Returns:
            Recommended position size in units
        """
        # Method 1: Fixed fraction risk
        risk_per_trade = self.portfolio_value * self.risk_config.max_position_size
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk > 0:
            fixed_fraction_size = risk_per_trade / price_risk
        else:
            fixed_fraction_size = 0

        # Method 2: Volatility-adjusted sizing
        if volatility:
            target_risk = 0.02  # 2% portfolio risk
            vol_adjusted_size = (self.portfolio_value * target_risk) / (entry_price * volatility)
        else:
            vol_adjusted_size = fixed_fraction_size

        # Method 3: Kelly Criterion (simplified)
        # Requires win rate and avg win/loss - using conservative estimate
        kelly_fraction = 0.25 * self.risk_config.max_position_size
        kelly_size = (self.portfolio_value * kelly_fraction) / entry_price

        # Take the minimum of all methods for safety
        optimal_size = min(fixed_fraction_size, vol_adjusted_size, kelly_size)

        # Apply leverage limits
        max_leverage = self.risk_config.max_leverage.get(market.value, 1.0)
        max_position_value = self.portfolio_value * max_leverage * self.risk_config.max_position_size
        max_size = max_position_value / entry_price

        final_size = min(optimal_size, max_size)

        logger.info(f"Position size for {symbol}: {final_size:.4f} units "
                   f"(value: ${final_size * entry_price:.2f})")

        return final_size

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: float = None,
        support_resistance: float = None,
        volatility: float = None
    ) -> float:
        """
        Calculate stop loss price using multiple methods.

        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            atr: Average True Range
            support_resistance: Nearby support/resistance level
            volatility: Asset volatility

        Returns:
            Stop loss price
        """
        stops = []

        # ATR-based stop
        if atr:
            atr_stop = atr * self.risk_config.stop_loss_atr_multiplier
            if side == 'long':
                stops.append(entry_price - atr_stop)
            else:
                stops.append(entry_price + atr_stop)

        # Percentage-based stop (2%)
        pct_stop = entry_price * 0.02
        if side == 'long':
            stops.append(entry_price - pct_stop)
        else:
            stops.append(entry_price + pct_stop)

        # Volatility-based stop
        if volatility:
            vol_stop = entry_price * volatility * 2
            if side == 'long':
                stops.append(entry_price - vol_stop)
            else:
                stops.append(entry_price + vol_stop)

        # Support/Resistance based stop
        if support_resistance:
            buffer = entry_price * 0.002  # 0.2% buffer
            if side == 'long':
                stops.append(support_resistance - buffer)
            else:
                stops.append(support_resistance + buffer)

        # Select stop loss
        if side == 'long':
            stop_loss = max(stops) if stops else entry_price * 0.98
        else:
            stop_loss = min(stops) if stops else entry_price * 1.02

        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: 'long' or 'short'
            risk_reward_ratio: Target risk-reward ratio

        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio

        if side == 'long':
            return entry_price + reward
        else:
            return entry_price - reward

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        holding_period: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Expected Shortfall.

        Args:
            returns: Historical returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            holding_period: Holding period in days

        Returns:
            Tuple of (VaR, Expected Shortfall)
        """
        if len(returns) < 30:
            return 0.0, 0.0

        # Historical VaR
        var_pct = np.percentile(returns, (1 - confidence) * 100)

        # Scale for holding period
        var_scaled = var_pct * np.sqrt(holding_period)

        # Expected Shortfall (CVaR)
        tail_returns = returns[returns <= var_pct]
        es = tail_returns.mean() if len(tail_returns) > 0 else var_pct

        return var_scaled, es

    def calculate_portfolio_var(
        self,
        position_returns: Dict[str, pd.Series],
        position_weights: Dict[str, float],
        confidence: float = 0.99
    ) -> float:
        """
        Calculate portfolio VaR considering correlations.

        Args:
            position_returns: Dict of returns series per position
            position_weights: Dict of portfolio weights
            confidence: Confidence level

        Returns:
            Portfolio VaR
        """
        if not position_returns:
            return 0.0

        # Build returns matrix
        returns_df = pd.DataFrame(position_returns)
        weights = np.array([position_weights.get(col, 0) for col in returns_df.columns])

        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        var, _ = self.calculate_var(portfolio_returns, confidence)
        return var * self.portfolio_value

    def register_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_price: float = None
    ) -> PositionRisk:
        """Register a new position for risk monitoring"""
        current_price = current_price or entry_price

        # Calculate unrealized PnL
        if side == 'long':
            unrealized_pnl = (current_price - entry_price) * size
        else:
            unrealized_pnl = (entry_price - current_price) * size

        # Calculate risk amount
        risk_amount = abs(entry_price - stop_loss) * size

        # Calculate risk percentage
        risk_pct = risk_amount / self.portfolio_value

        position_risk = PositionRisk(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            unrealized_pnl=unrealized_pnl,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            var_95=0.0,  # To be calculated with historical data
            expected_shortfall=0.0
        )

        self.positions[symbol] = position_risk
        return position_risk

    def update_position(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[PositionRisk]:
        """Update position with current price"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pos.current_price = current_price

        if pos.side == 'long':
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size

        return pos

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]

        if pos.side == 'long' and current_price <= pos.stop_loss:
            logger.warning(f"Stop loss triggered for {symbol} at {current_price}")
            return True
        elif pos.side == 'short' and current_price >= pos.stop_loss:
            logger.warning(f"Stop loss triggered for {symbol} at {current_price}")
            return True

        return False

    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if take profit is triggered"""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]

        if pos.side == 'long' and current_price >= pos.take_profit:
            logger.info(f"Take profit triggered for {symbol} at {current_price}")
            return True
        elif pos.side == 'short' and current_price <= pos.take_profit:
            logger.info(f"Take profit triggered for {symbol} at {current_price}")
            return True

        return False

    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current portfolio risk metrics"""
        metrics = RiskMetrics()

        for pos in self.positions.values():
            position_value = pos.size * pos.current_price

            if pos.side == 'long':
                metrics.long_exposure += position_value
            else:
                metrics.short_exposure += position_value

            metrics.positions_at_risk += 1

        metrics.total_exposure = metrics.long_exposure + metrics.short_exposure
        metrics.net_exposure = metrics.long_exposure - metrics.short_exposure

        # Calculate current drawdown
        current_equity = self.portfolio_value + sum(
            p.unrealized_pnl for p in self.positions.values()
        )
        metrics.current_drawdown = (self.equity_high - current_equity) / self.equity_high

        # Update equity high
        if current_equity > self.equity_high:
            self.equity_high = current_equity

        return metrics

    def check_risk_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check all risk limits and return whether trading should continue.

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        metrics = self.get_risk_metrics()

        # Check max drawdown
        if metrics.current_drawdown >= self.risk_config.max_drawdown:
            return False, f"Max drawdown exceeded: {metrics.current_drawdown:.2%}"

        # Check daily loss
        if self.daily_pnl_history:
            daily_pnl_pct = self.daily_pnl_history[-1] / self.portfolio_value
            if daily_pnl_pct <= -self.risk_config.max_daily_loss:
                return False, f"Max daily loss exceeded: {daily_pnl_pct:.2%}"

        # Check total portfolio risk
        total_risk_pct = sum(p.risk_pct for p in self.positions.values())
        if total_risk_pct > self.risk_config.max_portfolio_risk:
            return False, f"Max portfolio risk exceeded: {total_risk_pct:.2%}"

        return True, None

    def can_open_position(
        self,
        symbol: str,
        position_value: float,
        risk_amount: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new position can be opened within risk limits.

        Returns:
            Tuple of (can_open, reason_if_not)
        """
        # Check if trading is halted
        if self.is_trading_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # Check position size limit
        if position_value > self.portfolio_value * self.risk_config.max_position_size * 2:
            return False, "Position size exceeds limit"

        # Check total exposure
        metrics = self.get_risk_metrics()
        new_total_exposure = metrics.total_exposure + position_value

        if new_total_exposure > self.portfolio_value * 2:
            return False, "Total exposure limit exceeded"

        # Check total risk
        new_risk_pct = risk_amount / self.portfolio_value
        total_risk = sum(p.risk_pct for p in self.positions.values()) + new_risk_pct

        if total_risk > self.risk_config.max_portfolio_risk:
            return False, f"Adding position would exceed portfolio risk limit"

        return True, None

    def close_position(self, symbol: str) -> Optional[PositionRisk]:
        """Remove position from tracking"""
        return self.positions.pop(symbol, None)

    def update_daily_pnl(self, pnl: float):
        """Record daily PnL"""
        self.daily_pnl_history.append(pnl)

        # Keep only last 252 trading days
        if len(self.daily_pnl_history) > 252:
            self.daily_pnl_history = self.daily_pnl_history[-252:]

    def get_correlation_risk(
        self,
        returns_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Calculate correlation matrix for positions"""
        if not returns_data:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()

        # Flag high correlations
        high_corr = (correlation_matrix.abs() > self.risk_config.max_correlation)
        np.fill_diagonal(high_corr.values, False)

        if high_corr.any().any():
            logger.warning("High correlation detected between positions")

        return correlation_matrix

    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        metrics = self.get_risk_metrics()

        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'equity_high': self.equity_high,
            'metrics': {
                'total_exposure': metrics.total_exposure,
                'net_exposure': metrics.net_exposure,
                'current_drawdown': metrics.current_drawdown,
                'positions_at_risk': metrics.positions_at_risk
            },
            'positions': [
                {
                    'symbol': p.symbol,
                    'side': p.side,
                    'size': p.size,
                    'unrealized_pnl': p.unrealized_pnl,
                    'risk_pct': p.risk_pct
                }
                for p in self.positions.values()
            ],
            'limits': {
                'max_drawdown': self.risk_config.max_drawdown,
                'max_daily_loss': self.risk_config.max_daily_loss,
                'max_position_size': self.risk_config.max_position_size
            },
            'status': 'HALTED' if self.is_trading_halted else 'ACTIVE'
        }

        return report
