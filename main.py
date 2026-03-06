"""
Arbitrage & Risk Monitoring Engine – Main Entry Point
=====================================================
Orchestrates the full monitoring pipeline:

  1. Load configuration
  2. Initialise exchange connectors
  3. For each symbol pair on each exchange combination:
       a. Fetch ticker + order book (with caching)
       b. Calculate spread (buy cheapest, sell most expensive)
       c. Evaluate risk on both sides
       d. Score the opportunity
       e. If viable, call AI explainer and log/alert
  4. Sleep until next poll cycle; repeat indefinitely

Run:  python main.py
      python main.py --config config/settings.yaml
      python main.py --symbols BTC/USDT ETH/USDT --poll 10
"""

import argparse
import asyncio
import itertools
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Path setup so sub-packages resolve correctly when run from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from ai.opportunity_explainer import OpportunityExplainer
from connectors.base_exchange import BaseExchange, FeeSchedule, OrderBook, Ticker
from connectors.binance_connector import BinanceConnector
from connectors.kraken_connector import KrakenConnector
from core.opportunity_engine import ArbitrageOpportunity, OpportunityEngine
from core.risk_model import RiskModel
from core.spread_engine import SpreadEngine, SpreadResult
from data.market_cache import MarketDataCache

logger = logging.getLogger("arbitrage_engine")


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Environment variable overrides
    cfg["exchanges"]["binance"]["api_key"] = os.environ.get(
        "BINANCE_API_KEY", cfg["exchanges"]["binance"].get("api_key", ""))
    cfg["exchanges"]["binance"]["api_secret"] = os.environ.get(
        "BINANCE_API_SECRET", cfg["exchanges"]["binance"].get("api_secret", ""))
    cfg["exchanges"]["kraken"]["api_key"] = os.environ.get(
        "KRAKEN_API_KEY", cfg["exchanges"]["kraken"].get("api_key", ""))
    cfg["exchanges"]["kraken"]["api_secret"] = os.environ.get(
        "KRAKEN_API_SECRET", cfg["exchanges"]["kraken"].get("api_secret", ""))
    cfg["ai"]["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    return cfg


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format",
                      "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log_file = log_cfg.get("file", "logs/arbitrage_engine.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.get("max_bytes", 10_485_760),
        backupCount=log_cfg.get("backup_count", 5),
    )
    handlers.append(file_handler)

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ArbitrageEngine:
    """
    Top-level orchestrator.  Owns all components and drives the main loop.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cache = MarketDataCache()

        trade_size = cfg["trade"]["default_size_usd"]
        self.spread_engine = SpreadEngine(trade_size_usd=trade_size)
        self.risk_model = RiskModel(
            depth_band_pct=cfg["risk"]["depth_band_pct"],
            trade_size_usd=trade_size,
            weights=cfg["risk"]["weights"],
            price_history_len=cfg["risk"]["price_history_len"],
        )
        self.opp_engine = OpportunityEngine(
            min_spread_pct=cfg["scoring"]["min_spread_pct"],
            target_spread_pct=cfg["scoring"]["target_spread_pct"],
            execution_prob_threshold=cfg["scoring"]["execution_prob_threshold"],
            weights=cfg["scoring"]["weights"],
        )
        self.explainer = OpportunityExplainer(
            model=cfg["ai"]["model"],
            max_tokens=cfg["ai"]["max_tokens"],
        ) if cfg["ai"]["enabled"] else None

        self.exchanges: list[BaseExchange] = self._init_exchanges()
        self.symbols: list[str] = cfg["symbols"]
        self._cooldown: dict[str, float] = {}   # key → last_alert_time
        self._cooldown_secs = cfg["monitor"]["opportunity_cooldown_seconds"]
        self._error_counts: dict[str, int] = {}

        # Stats
        self._cycles = 0
        self._opportunities_detected = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Exchange initialisation
    # ------------------------------------------------------------------

    def _init_exchanges(self) -> list[BaseExchange]:
        exchanges = []
        ex_cfg = self.cfg["exchanges"]

        if ex_cfg.get("binance", {}).get("enabled"):
            b = ex_cfg["binance"]
            exchanges.append(BinanceConnector(
                api_key=b.get("api_key", ""),
                api_secret=b.get("api_secret", ""),
                sandbox=b.get("sandbox", False),
            ))
            logger.info("Initialised Binance connector")

        if ex_cfg.get("kraken", {}).get("enabled"):
            k = ex_cfg["kraken"]
            exchanges.append(KrakenConnector(
                api_key=k.get("api_key", ""),
                api_secret=k.get("api_secret", ""),
                sandbox=k.get("sandbox", False),
            ))
            logger.info("Initialised Kraken connector")

        return exchanges

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("=" * 60)
        logger.info("  Arbitrage & Risk Monitoring Engine  –  STARTING")
        logger.info("  Exchanges : %s",
                    ", ".join(e.name for e in self.exchanges))
        logger.info("  Symbols   : %s", ", ".join(self.symbols))
        logger.info("=" * 60)

        poll = self.cfg["monitor"]["poll_interval_seconds"]

        while True:
            cycle_start = time.time()
            self._cycles += 1
            logger.info("── Cycle %d ──────────────────────────────", self._cycles)

            tasks = []
            for symbol in self.symbols:
                for buy_ex, sell_ex in itertools.permutations(
                        self.exchanges, 2):
                    tasks.append(
                        self._evaluate_pair(symbol, buy_ex, sell_ex))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    logger.error("Task error: %s", r)

            # Periodic cache cleanup
            purged = await self.cache.purge_expired()
            if purged:
                logger.debug("Cache: purged %d expired entries", purged)

            elapsed = time.time() - cycle_start
            sleep_time = max(0.0, poll - elapsed)
            logger.info(
                "Cycle %d done in %.2fs (sleep %.1fs) | "
                "total opps detected: %d",
                self._cycles, elapsed, sleep_time,
                self._opportunities_detected,
            )
            await asyncio.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Per-pair evaluation
    # ------------------------------------------------------------------

    async def _evaluate_pair(
        self,
        symbol: str,
        buy_ex: BaseExchange,
        sell_ex: BaseExchange,
    ) -> None:
        """Fetch data, calculate spread, score opportunity for one pair."""
        key = f"{buy_ex.name}→{sell_ex.name}:{symbol}"

        try:
            buy_ticker, buy_ob, buy_fees = await self._fetch_market_data(
                buy_ex, symbol)
            sell_ticker, sell_ob, sell_fees = await self._fetch_market_data(
                sell_ex, symbol)
        except Exception as exc:
            self._error_counts[buy_ex.name] = (
                self._error_counts.get(buy_ex.name, 0) + 1)
            logger.warning("Data fetch failed for %s: %s", key, exc)
            return

        # Only proceed if buy ask < sell bid (potential spread)
        if buy_ticker.ask >= sell_ticker.bid:
            logger.debug("No spread: %s  (buy_ask=%.4f >= sell_bid=%.4f)",
                         key, buy_ticker.ask, sell_ticker.bid)
            return

        # Spread calculation
        spread = self.spread_engine.calculate(
            buy_ticker, buy_ob, buy_fees,
            sell_ticker, sell_ob, sell_fees,
        )

        if not spread.is_profitable:
            logger.debug("Not profitable after costs: %s  (%.4f%%)",
                         key, spread.slippage_adjusted_spread_pct)
            return

        # Risk evaluation
        buy_risk = self.risk_model.evaluate(buy_ob, buy_ticker)
        sell_risk = self.risk_model.evaluate(sell_ob, sell_ticker)

        # Opportunity scoring
        opp = self.opp_engine.evaluate(spread, buy_risk, sell_risk)

        if opp.tier == "NONE":
            return

        self._opportunities_detected += 1
        self._log_opportunity(opp)

        # AI explanation (with cooldown guard)
        await self._maybe_explain(opp)

    # ------------------------------------------------------------------
    # Data fetching (with cache)
    # ------------------------------------------------------------------

    async def _fetch_market_data(
        self,
        exchange: BaseExchange,
        symbol: str,
    ) -> tuple[Ticker, OrderBook, FeeSchedule]:
        """Fetch ticker, order book, and fees with TTL caching."""

        ticker = await self.cache.get_ticker(exchange.name, symbol)
        if ticker is None:
            ticker = await exchange.get_ticker(symbol)
            await self.cache.set_ticker(exchange.name, symbol, ticker)

        ob = await self.cache.get_orderbook(exchange.name, symbol)
        if ob is None:
            ob = await exchange.get_orderbook(symbol, depth=20)
            await self.cache.set_orderbook(exchange.name, symbol, ob)

        fees = await self.cache.get_fees(exchange.name, symbol)
        if fees is None:
            fees = await exchange.get_trading_fees(symbol)
            await self.cache.set_fees(exchange.name, symbol, fees)

        return ticker, ob, fees

    # ------------------------------------------------------------------
    # Logging / alerting
    # ------------------------------------------------------------------

    def _log_opportunity(self, opp: ArbitrageOpportunity) -> None:
        d = opp.to_dict()
        logger.info(
            "\n╔══════════════════════════════════════════════════╗\n"
            "║  ARBITRAGE OPPORTUNITY DETECTED  [%s]\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Symbol   : %-36s ║\n"
            "║  Route    : %-12s  →  %-20s ║\n"
            "║  Raw Sprd : %-6.4f%%   Fee Adj: %-6.4f%%           ║\n"
            "║  Net Sprd : %-6.4f%%   Risk   : %-6.3f             ║\n"
            "║  Exec Prob: %-6.2f%%   Score  : %-6.1f/100         ║\n"
            "╚══════════════════════════════════════════════════╝",
            opp.tier,
            opp.symbol,
            opp.buy_exchange.upper(), opp.sell_exchange.upper(),
            opp.raw_spread_pct, opp.fee_adjusted_spread_pct,
            opp.slippage_adjusted_spread_pct, opp.combined_risk_score,
            opp.execution_probability * 100, opp.opportunity_score,
        )

    async def _maybe_explain(self, opp: ArbitrageOpportunity) -> None:
        """Rate-limited AI explanation call."""
        if self.explainer is None:
            return

        min_tier = self.cfg["ai"].get("min_tier_for_explanation", "GOOD")
        tier_rank = {"NONE": 0, "MARGINAL": 1, "GOOD": 2, "EXCELLENT": 3}
        if tier_rank.get(opp.tier, 0) < tier_rank.get(min_tier, 2):
            return

        cooldown_key = (f"{opp.buy_exchange}:{opp.sell_exchange}"
                        f":{opp.symbol}")
        last = self._cooldown.get(cooldown_key, 0)
        if time.time() - last < self._cooldown_secs:
            logger.debug("AI explanation on cooldown for %s", cooldown_key)
            return

        self._cooldown[cooldown_key] = time.time()
        try:
            result = await self.explainer.explain(opp)
            logger.info(
                "\n── AI ANALYSIS ─────────────────────────────────────\n"
                "%s\n"
                "  → Action    : %s\n"
                "  → Confidence: %s\n"
                "────────────────────────────────────────────────────",
                result.explanation,
                result.recommended_action,
                result.confidence,
            )
        except Exception as exc:
            logger.warning("AI explanation failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crypto Arbitrage & Risk Monitoring Engine")
    p.add_argument("--config", default="config/settings.yaml",
                   help="Path to settings.yaml")
    p.add_argument("--symbols", nargs="+",
                   help="Override symbols list from config")
    p.add_argument("--poll", type=float,
                   help="Override poll interval (seconds)")
    p.add_argument("--no-ai", action="store_true",
                   help="Disable AI explanation layer")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.symbols:
        cfg["symbols"] = args.symbols
    if args.poll:
        cfg["monitor"]["poll_interval_seconds"] = args.poll
    if args.no_ai:
        cfg["ai"]["enabled"] = False

    engine = ArbitrageEngine(cfg)
    try:
        await engine.run()
    except KeyboardInterrupt:
        uptime = time.time() - engine._start_time
        logger.info(
            "Engine stopped. Cycles: %d | Opportunities: %d | Uptime: %.0fs",
            engine._cycles, engine._opportunities_detected, uptime,
        )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
