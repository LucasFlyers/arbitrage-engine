"""
Spread Calculation Engine
=========================
Calculates raw, percentage, fee-adjusted, and slippage-adjusted spreads
between two exchanges for a given trading pair.

All spread values are expressed in quote-currency units unless noted.
"""

import logging
from dataclasses import dataclass

from connectors.base_exchange import FeeSchedule, OrderBook, Ticker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SpreadResult:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float                # best ask on buy side
    sell_price: float               # best bid on sell side
    raw_spread: float               # sell_price - buy_price
    spread_pct: float               # raw_spread / buy_price * 100
    fee_cost_pct: float             # total fee drag as % of buy_price
    fee_adjusted_spread: float      # raw_spread minus fee costs
    fee_adjusted_spread_pct: float
    slippage_adjusted_spread: float
    slippage_adjusted_spread_pct: float
    is_profitable: bool             # slippage_adjusted_spread > 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "buy_price": round(self.buy_price, 6),
            "sell_price": round(self.sell_price, 6),
            "raw_spread": round(self.raw_spread, 6),
            "spread_pct": round(self.spread_pct, 4),
            "fee_cost_pct": round(self.fee_cost_pct, 4),
            "fee_adjusted_spread": round(self.fee_adjusted_spread, 6),
            "fee_adjusted_spread_pct": round(self.fee_adjusted_spread_pct, 4),
            "slippage_adjusted_spread": round(self.slippage_adjusted_spread, 6),
            "slippage_adjusted_spread_pct": round(
                self.slippage_adjusted_spread_pct, 4),
            "is_profitable": self.is_profitable,
        }


# ---------------------------------------------------------------------------
# Slippage Simulator
# ---------------------------------------------------------------------------

class SlippageSimulator:
    """
    Simulates the average execution price when filling a given notional
    value against a live order book.

    Works by walking the book level-by-level until the entire trade size
    is consumed, then computing the volume-weighted average price (VWAP).
    """

    def simulate_buy(self, orderbook: OrderBook, notional_usd: float) -> float:
        """
        Walk the ask side of the book.
        Returns the VWAP execution price for buying `notional_usd` worth.
        """
        return self._walk_book(
            levels=orderbook.asks,
            notional=notional_usd,
            is_buy=True,
        )

    def simulate_sell(self, orderbook: OrderBook, notional_usd: float) -> float:
        """
        Walk the bid side of the book.
        Returns the VWAP execution price for selling `notional_usd` worth.
        """
        return self._walk_book(
            levels=orderbook.bids,
            notional=notional_usd,
            is_buy=False,
        )

    def _walk_book(self, levels: list[list[float]],
                   notional: float, is_buy: bool) -> float:
        """
        Walk order book levels to compute VWAP for filling `notional` USD.

        Parameters
        ----------
        levels   : [[price, qty], ...]  asks (ascending) or bids (descending)
        notional : USD value to fill
        is_buy   : True=consuming asks, False=consuming bids
        """
        remaining = notional
        total_cost = 0.0
        total_qty = 0.0

        for price, qty in levels:
            level_value = price * qty
            if remaining <= 0:
                break
            if level_value >= remaining:
                filled_qty = remaining / price
                total_cost += remaining
                total_qty += filled_qty
                remaining = 0
            else:
                total_cost += level_value
                total_qty += qty
                remaining -= level_value

        if total_qty == 0:
            return levels[0][0] if levels else 0.0

        vwap = total_cost / total_qty

        if remaining > 0:
            # Could not fully fill — warn and use the last price with penalty
            last_price = levels[-1][0] if levels else vwap
            penalty = last_price * 0.01 * (1 if is_buy else -1)
            logger.debug(
                "Order book too thin – %.2f USD unfilled, applying penalty",
                remaining)
            vwap = last_price + penalty

        return vwap

    def estimate_slippage_pct(self, orderbook: OrderBook,
                              notional_usd: float,
                              is_buy: bool) -> float:
        """
        Returns slippage as a percentage of the top-of-book price.
        Positive = worse than best price.
        """
        if is_buy:
            top_price = orderbook.best_ask
            exec_price = self.simulate_buy(orderbook, notional_usd)
            return (exec_price - top_price) / top_price * 100
        else:
            top_price = orderbook.best_bid
            exec_price = self.simulate_sell(orderbook, notional_usd)
            return (top_price - exec_price) / top_price * 100


# ---------------------------------------------------------------------------
# Spread Engine
# ---------------------------------------------------------------------------

class SpreadEngine:
    """
    Calculates all spread metrics for a cross-exchange arbitrage opportunity.

    Usage
    -----
    engine = SpreadEngine(trade_size_usd=5000)
    result = engine.calculate(
        buy_ticker, buy_orderbook, buy_fees,
        sell_ticker, sell_orderbook, sell_fees,
    )
    """

    def __init__(self, trade_size_usd: float = 5_000.0):
        self.trade_size_usd = trade_size_usd
        self._slippage = SlippageSimulator()

    def calculate(
        self,
        buy_ticker: Ticker,
        buy_orderbook: OrderBook,
        buy_fees: FeeSchedule,
        sell_ticker: Ticker,
        sell_orderbook: OrderBook,
        sell_fees: FeeSchedule,
    ) -> SpreadResult:
        """
        Compute a full SpreadResult between two exchanges.

        Convention:
          - We BUY on the cheaper exchange (lower ask price).
          - We SELL on the more expensive exchange (higher bid price).
        The caller is responsible for passing the correct buy/sell pair.
        """
        buy_price = buy_ticker.ask
        sell_price = sell_ticker.bid

        # ---------------------------------------------------------------
        # 1.  Raw spread
        # ---------------------------------------------------------------
        raw_spread = sell_price - buy_price
        spread_pct = (raw_spread / buy_price) * 100 if buy_price > 0 else 0.0

        # ---------------------------------------------------------------
        # 2.  Fee adjustment
        #     Cost = taker fee on buy + taker fee on sell + withdrawal fee
        #     Expressed as percentage of buy_price.
        # ---------------------------------------------------------------
        buy_fee_pct = buy_fees.taker_fee_pct / 100         # e.g. 0.001
        sell_fee_pct = sell_fees.taker_fee_pct / 100
        withdrawal_cost_pct = (
            buy_fees.withdrawal_fee_flat / buy_price
            + buy_fees.withdrawal_fee_pct / 100
        ) if buy_price > 0 else 0.0

        fee_cost_pct = (buy_fee_pct + sell_fee_pct + withdrawal_cost_pct) * 100
        fee_drag_abs = buy_price * (buy_fee_pct + sell_fee_pct +
                                    withdrawal_cost_pct)

        fee_adjusted_spread = raw_spread - fee_drag_abs
        fee_adjusted_spread_pct = (
            fee_adjusted_spread / buy_price * 100 if buy_price > 0 else 0.0
        )

        # ---------------------------------------------------------------
        # 3.  Slippage adjustment
        # ---------------------------------------------------------------
        buy_slippage_pct = self._slippage.estimate_slippage_pct(
            buy_orderbook, self.trade_size_usd, is_buy=True)
        sell_slippage_pct = self._slippage.estimate_slippage_pct(
            sell_orderbook, self.trade_size_usd, is_buy=False)

        total_slippage_pct = buy_slippage_pct + sell_slippage_pct
        slippage_drag_abs = buy_price * total_slippage_pct / 100

        slippage_adjusted_spread = fee_adjusted_spread - slippage_drag_abs
        slippage_adjusted_spread_pct = (
            slippage_adjusted_spread / buy_price * 100
            if buy_price > 0 else 0.0
        )

        logger.debug(
            "[%s] %s→%s  raw=%.4f%%  fee_adj=%.4f%%  slip_adj=%.4f%%",
            buy_ticker.symbol,
            buy_ticker.exchange, sell_ticker.exchange,
            spread_pct, fee_adjusted_spread_pct, slippage_adjusted_spread_pct,
        )

        return SpreadResult(
            symbol=buy_ticker.symbol,
            buy_exchange=buy_ticker.exchange,
            sell_exchange=sell_ticker.exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            raw_spread=raw_spread,
            spread_pct=spread_pct,
            fee_cost_pct=fee_cost_pct,
            fee_adjusted_spread=fee_adjusted_spread,
            fee_adjusted_spread_pct=fee_adjusted_spread_pct,
            slippage_adjusted_spread=slippage_adjusted_spread,
            slippage_adjusted_spread_pct=slippage_adjusted_spread_pct,
            is_profitable=slippage_adjusted_spread > 0,
        )
