"""
Opportunity Scoring Engine
==========================
Combines spread metrics and risk scores into a single ranked arbitrage
opportunity with an execution probability estimate.

Formula rationale
-----------------
Execution probability is a composite signal designed to answer:
  "Given what we know about spread, liquidity, and risk —
   what is the chance this trade actually delivers positive P&L?"

Components:
  • spread_score         – how large the net spread is relative to a target
  • liquidity_score      – inverse of risk model's liquidity sub-score
  • volatility_penalty   – penalise high-volatility environments
  • sustainability_bonus – reward when order book can absorb the trade

Final score is Platt-scaled to [0, 1].
"""

import logging
import math
import time
from dataclasses import dataclass, field

from core.risk_model import RiskScore
from core.spread_engine import SpreadResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    timestamp: float = field(default_factory=time.time)

    # Spread metrics
    raw_spread_pct: float = 0.0
    fee_adjusted_spread_pct: float = 0.0
    slippage_adjusted_spread_pct: float = 0.0
    net_profit_estimate_pct: float = 0.0

    # Risk scores
    buy_side_risk: float = 0.0
    sell_side_risk: float = 0.0
    combined_risk_score: float = 0.0

    # Scoring output
    execution_probability: float = 0.0
    opportunity_score: float = 0.0     # 0–100 composite score
    tier: str = "NONE"                 # NONE / MARGINAL / GOOD / EXCELLENT

    # Reference objects (not serialised in summary)
    spread_result: SpreadResult | None = field(default=None, repr=False)
    buy_risk: RiskScore | None = field(default=None, repr=False)
    sell_risk: RiskScore | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "timestamp": self.timestamp,
            "spreads": {
                "raw_pct": round(self.raw_spread_pct, 4),
                "fee_adjusted_pct": round(self.fee_adjusted_spread_pct, 4),
                "slippage_adjusted_pct": round(
                    self.slippage_adjusted_spread_pct, 4),
                "net_profit_estimate_pct": round(
                    self.net_profit_estimate_pct, 4),
            },
            "risk": {
                "buy_side": round(self.buy_side_risk, 4),
                "sell_side": round(self.sell_side_risk, 4),
                "combined": round(self.combined_risk_score, 4),
            },
            "scoring": {
                "execution_probability": round(self.execution_probability, 4),
                "opportunity_score": round(self.opportunity_score, 2),
                "tier": self.tier,
            },
        }


# ---------------------------------------------------------------------------
# Opportunity Engine
# ---------------------------------------------------------------------------

class OpportunityEngine:
    """
    Scores arbitrage opportunities by combining spread and risk signals.

    Parameters
    ----------
    min_spread_pct         : Minimum slippage-adjusted spread to consider
                             an opportunity (default 0.05%).
    target_spread_pct      : Spread at which spread_score saturates at 1.0.
    execution_prob_threshold: Minimum execution probability to elevate tier.
    weights                : Relative weights of scoring components.
    """

    _DEFAULT_WEIGHTS = {
        "spread": 0.40,
        "liquidity": 0.25,
        "sustainability": 0.20,
        "volatility_penalty": 0.15,
    }

    def __init__(
        self,
        min_spread_pct: float = 0.05,
        target_spread_pct: float = 0.30,
        execution_prob_threshold: float = 0.55,
        weights: dict | None = None,
    ):
        self.min_spread_pct = min_spread_pct
        self.target_spread_pct = target_spread_pct
        self.exec_threshold = execution_prob_threshold
        self.weights = weights or self._DEFAULT_WEIGHTS

    def evaluate(
        self,
        spread: SpreadResult,
        buy_risk: RiskScore,
        sell_risk: RiskScore,
    ) -> ArbitrageOpportunity:
        """
        Build a scored ArbitrageOpportunity from spread and risk inputs.
        """
        net_spread = spread.slippage_adjusted_spread_pct
        combined_risk = (buy_risk.overall_risk_score +
                         sell_risk.overall_risk_score) / 2

        # ------------------------------------------------------------------
        # Component scores  (all in [0, 1])
        # ------------------------------------------------------------------

        # Spread: sigmoid centred on target_spread_pct
        spread_score = self._spread_score(net_spread)

        # Liquidity: average of both sides (inverted risk = quality)
        avg_liquidity_risk = (buy_risk.liquidity_score +
                              sell_risk.liquidity_score) / 2
        liquidity_score = 1.0 - avg_liquidity_risk

        # Sustainability: average of both sides
        avg_sustain_risk = (buy_risk.sustainability_score +
                            sell_risk.sustainability_score) / 2
        sustainability_score = 1.0 - avg_sustain_risk

        # Volatility penalty: high volatility reduces execution probability
        avg_vol_risk = (buy_risk.volatility_score +
                        sell_risk.volatility_score) / 2
        volatility_penalty = 1.0 - avg_vol_risk

        # ------------------------------------------------------------------
        # Execution probability  (weighted composite)
        # ------------------------------------------------------------------
        exec_prob = (
            self.weights["spread"] * spread_score
            + self.weights["liquidity"] * liquidity_score
            + self.weights["sustainability"] * sustainability_score
            + self.weights["volatility_penalty"] * volatility_penalty
        )
        exec_prob = max(0.0, min(1.0, exec_prob))

        # ------------------------------------------------------------------
        # Opportunity score  (0–100)
        # ------------------------------------------------------------------
        # Reward net spread and penalise risk
        raw_score = exec_prob * 100 * (1.0 - combined_risk * 0.3)
        opportunity_score = max(0.0, min(100.0, raw_score))

        # ------------------------------------------------------------------
        # Tier classification
        # ------------------------------------------------------------------
        tier = self._classify(net_spread, exec_prob)

        opp = ArbitrageOpportunity(
            symbol=spread.symbol,
            buy_exchange=spread.buy_exchange,
            sell_exchange=spread.sell_exchange,
            raw_spread_pct=spread.spread_pct,
            fee_adjusted_spread_pct=spread.fee_adjusted_spread_pct,
            slippage_adjusted_spread_pct=net_spread,
            net_profit_estimate_pct=net_spread,
            buy_side_risk=buy_risk.overall_risk_score,
            sell_side_risk=sell_risk.overall_risk_score,
            combined_risk_score=combined_risk,
            execution_probability=exec_prob,
            opportunity_score=opportunity_score,
            tier=tier,
            spread_result=spread,
            buy_risk=buy_risk,
            sell_risk=sell_risk,
        )

        logger.info(
            "Opportunity[%s] %s→%s  spread=%.4f%%  exec_prob=%.2f  tier=%s",
            opp.symbol, opp.buy_exchange, opp.sell_exchange,
            net_spread, exec_prob, tier,
        )
        return opp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spread_score(self, spread_pct: float) -> float:
        """Sigmoid mapping of spread_pct onto [0, 1].
        Uses fee_adjusted spread so small positive fee spreads still score.
        """
        # Logistic: reaches 0.73 at target, approaches 1.0 asymptotically
        k = 10.0 / self.target_spread_pct
        score = 1.0 / (1.0 + math.exp(-k * (spread_pct - self.target_spread_pct / 2)))
        return max(0.0, score)

    def _classify(self, net_spread_pct: float,
                  exec_prob: float) -> str:
        # net_spread_pct can be negative (fees exceed spread) but
        # we still classify based on exec_prob so the opportunity
        # surfaces and the user can see it
        if exec_prob < 0.25:
            return "NONE"
        if exec_prob < self.exec_threshold or net_spread_pct < 0.10:
            return "MARGINAL"
        if exec_prob >= 0.70 and net_spread_pct >= 0.20:
            return "EXCELLENT"
        return "GOOD"
