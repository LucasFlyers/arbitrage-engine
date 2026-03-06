"""
Risk Model
==========
Market-microstructure risk evaluation for arbitrage opportunities.

Produces a composite risk score from four sub-models:
  1. Liquidity Depth      – how much volume sits near the mid price
  2. Order Book Imbalance – bid/ask volume asymmetry
  3. Volume Sustainability– whether trade size overwhelms available liquidity
  4. Volatility Risk      – short-term price instability proxy

Each sub-score is in [0, 1] where 0 = low risk and 1 = high risk.
"""

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from connectors.base_exchange import OrderBook, Ticker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class RiskScore:
    symbol: str
    exchange: str
    liquidity_score: float       # 0=deep book, 1=thin book
    imbalance_score: float       # 0=balanced, 1=severely skewed
    sustainability_score: float  # 0=trade fits, 1=trade too large
    volatility_score: float      # 0=stable, 1=volatile
    overall_risk_score: float    # weighted composite

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "liquidity_score": round(self.liquidity_score, 4),
            "imbalance_score": round(self.imbalance_score, 4),
            "sustainability_score": round(self.sustainability_score, 4),
            "volatility_score": round(self.volatility_score, 4),
            "overall_risk_score": round(self.overall_risk_score, 4),
        }

    @property
    def label(self) -> str:
        s = self.overall_risk_score
        if s < 0.30:
            return "LOW"
        if s < 0.55:
            return "MEDIUM"
        if s < 0.75:
            return "HIGH"
        return "EXTREME"


# ---------------------------------------------------------------------------
# Risk Model
# ---------------------------------------------------------------------------

class RiskModel:
    """
    Parameters
    ----------
    depth_band_pct    : Price band (%) around mid price for liquidity depth.
    trade_size_usd    : Intended notional trade size in USD.
    weights           : Sub-score weights; must sum to 1.0.
    price_history_len : Number of price samples kept for volatility calc.
    """

    _DEFAULT_WEIGHTS = {
        "liquidity": 0.30,
        "imbalance": 0.20,
        "sustainability": 0.30,
        "volatility": 0.20,
    }

    def __init__(
        self,
        depth_band_pct: float = 0.5,
        trade_size_usd: float = 5_000.0,
        weights: dict | None = None,
        price_history_len: int = 20,
    ):
        self.depth_band_pct = depth_band_pct
        self.trade_size_usd = trade_size_usd
        self.weights = weights or self._DEFAULT_WEIGHTS
        self._price_history: dict[str, list[float]] = {}   # key = exchange:symbol
        self._history_len = price_history_len

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, orderbook: OrderBook, ticker: Ticker) -> RiskScore:
        """Evaluate all risk dimensions and return a composite RiskScore."""
        self._record_price(orderbook.exchange, orderbook.symbol, ticker.mid)

        liq_score = self._liquidity_score(orderbook)
        imb_score = self._imbalance_score(orderbook)
        sus_score = self._sustainability_score(orderbook)
        vol_score = self._volatility_score(orderbook.exchange, orderbook.symbol)

        overall = (
            self.weights["liquidity"] * liq_score
            + self.weights["imbalance"] * imb_score
            + self.weights["sustainability"] * sus_score
            + self.weights["volatility"] * vol_score
        )
        overall = max(0.0, min(1.0, overall))

        rs = RiskScore(
            symbol=orderbook.symbol,
            exchange=orderbook.exchange,
            liquidity_score=liq_score,
            imbalance_score=imb_score,
            sustainability_score=sus_score,
            volatility_score=vol_score,
            overall_risk_score=overall,
        )
        logger.debug("Risk[%s@%s]: %s (%.3f)", rs.symbol, rs.exchange,
                     rs.label, rs.overall_risk_score)
        return rs

    # ------------------------------------------------------------------
    # Sub-score implementations
    # ------------------------------------------------------------------

    def _liquidity_score(self, ob: OrderBook) -> float:
        """
        Measures USD-denominated volume within `depth_band_pct`% of mid.
        Higher volume → lower risk.

        Thresholds are calibrated for a $5,000 trade size:
          > $500k  within band → score ≈ 0.0
          < $10k   within band → score ≈ 1.0
        """
        mid = ob.mid_price
        if mid <= 0:
            return 1.0

        band = mid * self.depth_band_pct / 100
        lo = mid - band
        hi = mid + band

        bid_vol_usd = sum(
            p * v for p, v in ob.bids if p >= lo
        )
        ask_vol_usd = sum(
            p * v for p, v in ob.asks if p <= hi
        )
        total_vol_usd = bid_vol_usd + ask_vol_usd

        # Sigmoid-like mapping: 0 → 1.0, 500k → ~0.0
        score = 1.0 / (1.0 + total_vol_usd / 50_000)
        return max(0.0, min(1.0, score))

    def _imbalance_score(self, ob: OrderBook) -> float:
        """
        Order book imbalance: ratio of bid to ask volume across all levels.

        Perfectly balanced (ratio ≈ 1) → score ≈ 0.0
        Very skewed (ratio < 0.3 or > 3) → score → 1.0
        """
        bid_vol = sum(v for _, v in ob.bids)
        ask_vol = sum(v for _, v in ob.asks)

        if ask_vol == 0 or bid_vol == 0:
            return 1.0

        ratio = bid_vol / ask_vol           # 1.0 = perfectly balanced
        # Distance from balance on log scale
        log_imbalance = abs(math.log(ratio))    # 0 = balanced
        # Map to [0, 1]; log_imbalance > 2.3 (~10x skew) → near 1
        score = min(1.0, log_imbalance / 2.3)
        return score

    def _sustainability_score(self, ob: OrderBook) -> float:
        """
        Can the order book sustain the intended trade size without severe
        price impact?

        We measure how much of the ask side (for buying) can be filled
        within a 1% price band from best ask.
        """
        best_ask = ob.best_ask
        if best_ask <= 0:
            return 1.0

        ceiling = best_ask * 1.01
        available_usd = sum(
            p * v for p, v in ob.asks if p <= ceiling
        )

        if available_usd == 0:
            return 1.0

        # How many times our trade size fits in available liquidity
        ratio = available_usd / self.trade_size_usd
        # ratio > 10 → very sustainable → score ≈ 0
        score = 1.0 / (1.0 + ratio / 2)
        return max(0.0, min(1.0, score))

    def _volatility_score(self, exchange: str, symbol: str) -> float:
        """
        Short-term price volatility measured as the coefficient of variation
        of recent mid-price samples.  High CV → high risk score.
        """
        key = f"{exchange}:{symbol}"
        history = self._price_history.get(key, [])

        if len(history) < 3:
            return 0.3    # neutral prior with limited data

        try:
            mean = statistics.mean(history)
            stdev = statistics.stdev(history)
            cv = stdev / mean if mean > 0 else 0.0
            # cv > 0.01 (1%) intraday → extremely volatile for arb purposes
            score = min(1.0, cv / 0.005)
            return score
        except statistics.StatisticsError:
            return 0.5

    # ------------------------------------------------------------------
    # Price history management
    # ------------------------------------------------------------------

    def _record_price(self, exchange: str, symbol: str, price: float):
        key = f"{exchange}:{symbol}"
        if key not in self._price_history:
            self._price_history[key] = []
        self._price_history[key].append(price)
        # Trim to rolling window
        if len(self._price_history[key]) > self._history_len:
            self._price_history[key] = self._price_history[key][
                -self._history_len:
            ]
