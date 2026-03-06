"""
AI Opportunity Explainer
========================
Uses the OpenAI Chat Completions API to generate a natural-language
explanation of a detected arbitrage opportunity.

The module builds a structured prompt from the numerical opportunity data
and returns a human-readable analysis covering:
  • Why the spread exists
  • Market microstructure context
  • Liquidity and execution risks
  • Recommended trade approach

Required environment variable:
  OPENAI_API_KEY   – your OpenAI secret key

Optional environment variable:
  OPENAI_BASE_URL  – override the API base URL (e.g. for Azure OpenAI or
                     any OpenAI-compatible proxy)
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class ExplanationResult:
    opportunity_id: str
    explanation: str
    confidence: str    # LOW / MEDIUM / HIGH
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
        }


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def _build_prompt(opp_dict: dict, buy_risk_dict: dict,
                  sell_risk_dict: dict) -> str:
    """Construct the structured prompt sent to the LLM."""

    spreads = opp_dict.get("spreads", {})
    risk = opp_dict.get("risk", {})
    scoring = opp_dict.get("scoring", {})

    context = f"""
You are a senior quantitative analyst reviewing a live cryptocurrency
arbitrage opportunity. Analyse the data below and provide a concise
but thorough explanation.

=== OPPORTUNITY SUMMARY ===
Symbol          : {opp_dict['symbol']}
Buy on          : {opp_dict['buy_exchange'].upper()}  at best ask
Sell on         : {opp_dict['sell_exchange'].upper()} at best bid
Timestamp       : {opp_dict.get('timestamp', 'N/A')}

=== SPREAD METRICS ===
Raw Spread         : {spreads.get('raw_pct', 0):.4f}%
Fee-Adjusted Spread: {spreads.get('fee_adjusted_pct', 0):.4f}%
Slippage-Adjusted  : {spreads.get('slippage_adjusted_pct', 0):.4f}%
Net Profit Est.    : {spreads.get('net_profit_estimate_pct', 0):.4f}%

=== RISK SCORES  (0=low risk, 1=high risk) ===
Buy-Side Risk      : {risk.get('buy_side', 0):.3f}
Sell-Side Risk     : {risk.get('sell_side', 0):.3f}
Combined Risk      : {risk.get('combined', 0):.3f}

=== BUY-SIDE ORDER BOOK RISK ===
Liquidity Score    : {buy_risk_dict.get('liquidity_score', 0):.3f}
Imbalance Score    : {buy_risk_dict.get('imbalance_score', 0):.3f}
Sustainability     : {buy_risk_dict.get('sustainability_score', 0):.3f}
Volatility         : {buy_risk_dict.get('volatility_score', 0):.3f}

=== SELL-SIDE ORDER BOOK RISK ===
Liquidity Score    : {sell_risk_dict.get('liquidity_score', 0):.3f}
Imbalance Score    : {sell_risk_dict.get('imbalance_score', 0):.3f}
Sustainability     : {sell_risk_dict.get('sustainability_score', 0):.3f}
Volatility         : {sell_risk_dict.get('volatility_score', 0):.3f}

=== SCORING ===
Execution Probability: {scoring.get('execution_probability', 0):.2%}
Opportunity Score    : {scoring.get('opportunity_score', 0):.1f}/100
Tier                 : {scoring.get('tier', 'NONE')}

=== YOUR TASK ===
Write a concise analysis (4–6 paragraphs) covering:

1. WHY THE SPREAD EXISTS – Explain the likely microstructure reasons
   (e.g. liquidity imbalance, geographic fragmentation, fee differences,
   different user bases, temporary order flow).

2. EXECUTION RISKS – Highlight the top 2-3 risks specific to this
   opportunity (slippage, thin book, volatility, latency).

3. RECOMMENDED APPROACH – Suggest optimal trade size (small/medium/large
   relative to book depth), timing considerations, and whether to use
   market or limit orders.

4. OVERALL VERDICT – One clear sentence: Is this worth pursuing?

Be precise, factual, and avoid padding. Use the numbers provided.
""".strip()

    return context


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class OpportunityExplainer:
    """
    Calls the OpenAI Chat Completions API to explain an arbitrage opportunity.

    Parameters
    ----------
    model      : OpenAI model to use (default: gpt-4o).
    max_tokens : Maximum tokens for the completion response.

    Environment variables
    ---------------------
    OPENAI_API_KEY   : Required. Your OpenAI secret key (sk-...).
    OPENAI_BASE_URL  : Optional. Override the API base URL, e.g. for
                       Azure OpenAI or any OpenAI-compatible proxy.
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1000,
    ):
        self.model = model
        self.max_tokens = max_tokens

    async def explain(
        self,
        opportunity,  # ArbitrageOpportunity – typed loosely to avoid circular import
    ) -> ExplanationResult:
        """
        Generate a natural-language explanation for the opportunity.
        Falls back to a rule-based explanation if the API call fails.
        """
        opp_dict = opportunity.to_dict()
        buy_risk_dict = (opportunity.buy_risk.to_dict()
                         if opportunity.buy_risk else {})
        sell_risk_dict = (opportunity.sell_risk.to_dict()
                          if opportunity.sell_risk else {})

        prompt = _build_prompt(opp_dict, buy_risk_dict, sell_risk_dict)
        op_id = (f"{opportunity.buy_exchange}_"
                 f"{opportunity.sell_exchange}_"
                 f"{opportunity.symbol.replace('/', '_')}")

        try:
            explanation = await self._call_api(prompt)
        except Exception as exc:
            logger.warning("OpenAI API call failed (%s) – using fallback", exc)
            explanation = self._fallback_explanation(opportunity)

        confidence = self._infer_confidence(opportunity)
        action = self._recommended_action(opportunity)

        return ExplanationResult(
            opportunity_id=op_id,
            explanation=explanation,
            confidence=confidence,
            recommended_action=action,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call_api(self, prompt: str) -> str:
        """Make an async call to the OpenAI Chat Completions endpoint."""
        import aiohttp

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before starting the engine:  "
                "export OPENAI_API_KEY='sk-...'"
            )

        base_url = os.environ.get("OPENAI_BASE_URL",
                                  self._DEFAULT_BASE_URL).rstrip("/")
        url = f"{base_url}/chat/completions"

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.3,          # lower = more factual / consistent
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior quantitative analyst specialising in "
                        "cryptocurrency market microstructure and arbitrage. "
                        "Be precise, data-driven, and concise."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload,
                                    headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    def _fallback_explanation(self, opp) -> str:
        """
        Rule-based template explanation used when the LLM is unavailable.
        """
        s = opp.spread_result
        risk = opp.combined_risk_score

        risk_label = "low" if risk < 0.4 else "moderate" if risk < 0.6 else "high"

        lines = [
            f"Arbitrage opportunity detected: {opp.symbol} "
            f"({opp.buy_exchange.upper()} → {opp.sell_exchange.upper()}).",
            "",
            f"The raw spread of {opp.raw_spread_pct:.3f}% narrows to "
            f"{opp.slippage_adjusted_spread_pct:.3f}% after accounting for "
            f"trading fees and estimated slippage.",
            "",
            f"Market risk is assessed as {risk_label} (composite score "
            f"{risk:.2f}/1.00). "
            f"Buy-side liquidity risk is {opp.buy_side_risk:.2f} and "
            f"sell-side liquidity risk is {opp.sell_side_risk:.2f}.",
            "",
            f"Execution probability: {opp.execution_probability:.0%}. "
            f"Opportunity tier: {opp.tier}.",
            "",
            "Recommendation: "
            + (
                "This spread is likely execution-viable at small trade sizes. "
                "Use market orders for speed and monitor for order book changes."
                if opp.execution_probability >= 0.60
                else "Proceed with caution. The spread is marginal after costs. "
                "Consider waiting for wider conditions or reducing trade size."
            ),
        ]
        return "\n".join(lines)

    def _infer_confidence(self, opp) -> str:
        ep = opp.execution_probability
        if ep >= 0.70:
            return "HIGH"
        if ep >= 0.50:
            return "MEDIUM"
        return "LOW"

    def _recommended_action(self, opp) -> str:
        tier = opp.tier
        if tier == "EXCELLENT":
            return "EXECUTE – Strong opportunity. Use market orders at medium size."
        if tier == "GOOD":
            return "CONSIDER – Viable opportunity. Verify live book before trading."
        if tier == "MARGINAL":
            return "MONITOR – Marginal. Wait for improved conditions."
        return "SKIP – Spread does not cover costs after risk adjustment."
