"""
Arbitrage Engine – Web Server
=============================
Runs the ArbitrageEngine in a background asyncio thread and exposes:

  GET  /               → serves dashboard HTML
  GET  /api/status     → engine status + uptime stats
  GET  /api/prices     → latest ticker snapshot for all symbols/exchanges
  GET  /api/opportunities  → last N detected opportunities
  GET  /stream         → Server-Sent Events (SSE) live feed

Run:
    python web_server.py
    python web_server.py --port 8080
"""

import argparse
import asyncio
import collections
import itertools
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Flask
from flask import Flask, Response, jsonify, send_from_directory

# ── project path ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from connectors.base_exchange import BaseExchange, FeeSchedule, OrderBook, Ticker
from connectors.binance_connector import BinanceConnector
from connectors.kraken_connector import KrakenConnector
from connectors.bybit_connector import BybitConnector
from connectors.okx_connector import OKXConnector
from core.opportunity_engine import ArbitrageOpportunity, OpportunityEngine
from core.risk_model import RiskModel
from core.spread_engine import SpreadEngine
from data.market_cache import MarketDataCache
from ai.opportunity_explainer import OpportunityExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web_server")

# ── Shared state (written by engine thread, read by Flask) ────────────────────

_state: dict[str, Any] = {
    "status": "starting",
    "cycles": 0,
    "uptime_seconds": 0,
    "start_time": time.time(),
    "opportunities_total": 0,
    "exchanges": [],
    "symbols": [],
    "prices": {},          # { "BTC/USDT": { "binance": {bid,ask,mid}, ... } }
    "opportunities": collections.deque(maxlen=50),   # recent opps (newest first)
    "last_cycle_ms": 0,
    "errors": 0,
}
_sse_clients: list[Any] = []   # SSE queue per connected browser tab
_state_lock = threading.Lock()


def _push_event(event_type: str, data: dict):
    """Broadcast an SSE event to all connected clients."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            _sse_clients.remove(q)
        except ValueError:
            pass


# ── Engine loop (runs in its own thread with its own event loop) ──────────────

class WebArbitrageEngine:
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
        self.explainer = (
            OpportunityExplainer(model=cfg["ai"]["model"],
                                 max_tokens=cfg["ai"]["max_tokens"])
            if cfg["ai"]["enabled"] else None
        )
        self.exchanges: list[BaseExchange] = self._init_exchanges()
        self.symbols: list[str] = cfg["symbols"]
        self._cooldown: dict[str, float] = {}
        self._cooldown_secs = cfg["monitor"]["opportunity_cooldown_seconds"]
        self._start_time = time.time()

    def _init_exchanges(self) -> list[BaseExchange]:
        exs = []
        ec = self.cfg["exchanges"]
        if ec.get("binance", {}).get("enabled"):
            b = ec["binance"]
            exs.append(BinanceConnector(b.get("api_key",""), b.get("api_secret",""), b.get("sandbox",False)))
            logger.info("Initialised Binance connector")
        if ec.get("kraken", {}).get("enabled"):
            k = ec["kraken"]
            exs.append(KrakenConnector(k.get("api_key",""), k.get("api_secret",""), k.get("sandbox",False)))
            logger.info("Initialised Kraken connector")
        if ec.get("bybit", {}).get("enabled"):
            b = ec["bybit"]
            exs.append(BybitConnector(b.get("api_key",""), b.get("api_secret",""), b.get("sandbox",False)))
            logger.info("Initialised Bybit connector")
        if ec.get("okx", {}).get("enabled"):
            o = ec["okx"]
            exs.append(OKXConnector(o.get("api_key",""), o.get("api_secret",""), o.get("sandbox",False)))
            logger.info("Initialised OKX connector")
        return exs

    async def run(self):
        with _state_lock:
            _state["exchanges"] = [e.name for e in self.exchanges]
            _state["symbols"] = self.symbols
            _state["status"] = "running"

        poll = self.cfg["monitor"]["poll_interval_seconds"]
        cycles = 0

        while True:
            t0 = time.time()
            cycles += 1

            tasks = []
            for symbol in self.symbols:
                for buy_ex, sell_ex in itertools.permutations(self.exchanges, 2):
                    tasks.append(self._evaluate_pair(symbol, buy_ex, sell_ex))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = sum(1 for r in results if isinstance(r, Exception))

            elapsed_ms = int((time.time() - t0) * 1000)
            with _state_lock:
                _state["cycles"] = cycles
                _state["uptime_seconds"] = int(time.time() - self._start_time)
                _state["last_cycle_ms"] = elapsed_ms
                _state["errors"] += errors

            _push_event("cycle", {
                "cycle": cycles,
                "elapsed_ms": elapsed_ms,
                "uptime": int(time.time() - self._start_time),
            })

            await self.cache.purge_expired()
            await asyncio.sleep(max(0, poll - (time.time() - t0)))

    async def _evaluate_pair(self, symbol, buy_ex, sell_ex):
        try:
            bt, bo, bf = await self._fetch(buy_ex, symbol)
            st, so, sf = await self._fetch(sell_ex, symbol)
        except Exception as e:
            raise

        # Update prices state
        with _state_lock:
            p = _state["prices"]
            if symbol not in p:
                p[symbol] = {}
            for ticker, ob in [(bt, bo), (st, so)]:
                p[symbol][ticker.exchange] = {
                    "bid": ticker.bid, "ask": ticker.ask,
                    "mid": round(ticker.mid, 2),
                    "volume_24h": ticker.volume_24h,
                    "timestamp": ticker.timestamp,
                }

        if bt.ask >= st.bid:
            return

        spread = self.spread_engine.calculate(bt, bo, bf, st, so, sf)
        if not spread.is_profitable:
            return

        buy_risk = self.risk_model.evaluate(bo, bt)
        sell_risk = self.risk_model.evaluate(so, st)
        opp = self.opp_engine.evaluate(spread, buy_risk, sell_risk)

        if opp.tier == "NONE":
            return

        opp_dict = opp.to_dict()

        with _state_lock:
            _state["opportunities_total"] += 1
            _state["opportunities"].appendleft(opp_dict)

        _push_event("opportunity", opp_dict)

        # AI explanation with cooldown
        ck = f"{opp.buy_exchange}:{opp.sell_exchange}:{opp.symbol}"
        if (self.explainer and
                time.time() - self._cooldown.get(ck, 0) > self._cooldown_secs):
            self._cooldown[ck] = time.time()
            try:
                result = await self.explainer.explain(opp)
                _push_event("explanation", {
                    "opportunity_id": result.opportunity_id,
                    "explanation": result.explanation,
                    "confidence": result.confidence,
                    "recommended_action": result.recommended_action,
                })
            except Exception as e:
                logger.warning("AI explanation failed: %s", e)

    async def _fetch(self, exchange, symbol):
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


def _run_engine(cfg):
    """Entry point for the engine background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    engine = WebArbitrageEngine(cfg)
    loop.run_until_complete(engine.run())


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="web/static")


@app.route("/")
def dashboard():
    return send_from_directory("web", "index.html")


@app.route("/api/status")
def api_status():
    with _state_lock:
        return jsonify({
            "status": _state["status"],
            "cycles": _state["cycles"],
            "uptime_seconds": _state["uptime_seconds"],
            "opportunities_total": _state["opportunities_total"],
            "exchanges": _state["exchanges"],
            "symbols": _state["symbols"],
            "last_cycle_ms": _state["last_cycle_ms"],
            "errors": _state["errors"],
        })


@app.route("/api/prices")
def api_prices():
    with _state_lock:
        return jsonify(dict(_state["prices"]))


@app.route("/api/opportunities")
def api_opportunities():
    with _state_lock:
        return jsonify(list(_state["opportunities"]))


@app.route("/stream")
def stream():
    import queue

    q: queue.Queue = queue.Queue(maxsize=100)
    _sse_clients.append(q)

    def generate():
        # Send initial heartbeat
        yield "event: connected\ndata: {}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield msg
                except Exception:
                    yield "event: heartbeat\ndata: {}\n\n"
        except GeneratorExit:
            pass
        finally:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Arbitrage Engine Web Server")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["ai"]["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    cfg["exchanges"]["binance"]["api_key"]    = os.environ.get("BINANCE_API_KEY", "")
    cfg["exchanges"]["binance"]["api_secret"] = os.environ.get("BINANCE_API_SECRET", "")
    cfg["exchanges"]["kraken"]["api_key"]     = os.environ.get("KRAKEN_API_KEY", "")
    cfg["exchanges"]["kraken"]["api_secret"]  = os.environ.get("KRAKEN_API_SECRET", "")
    cfg["exchanges"].setdefault("bybit", {})["api_key"]    = os.environ.get("BYBIT_API_KEY", "")
    cfg["exchanges"].setdefault("bybit", {})["api_secret"] = os.environ.get("BYBIT_API_SECRET", "")
    cfg["exchanges"].setdefault("okx", {})["api_key"]      = os.environ.get("OKX_API_KEY", "")
    cfg["exchanges"].setdefault("okx", {})["api_secret"]   = os.environ.get("OKX_API_SECRET", "")

    # Start engine in background thread
    t = threading.Thread(target=_run_engine, args=(cfg,), daemon=True)
    t.start()
    logger.info("Engine thread started")

    # Start Flask
    logger.info("Dashboard → http://localhost:%d", args.port)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
