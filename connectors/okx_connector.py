"""
OKX Exchange Connector
======================
Implements BaseExchange for the OKX REST API v5 (spot market).
OKX is accessible globally including Nigeria without a VPN.

API docs: https://www.okx.com/docs-v5/en/
Public endpoints require no API key.

Symbol format: OKX uses BTC-USDT (dash-separated) internally.
"""

import logging
import time
from typing import Optional

import aiohttp

from connectors.base_exchange import (
    BaseExchange, FeeSchedule, OrderBook, Ticker, TradingPair,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.okx.com"

_DEFAULT_MAKER_FEE = 0.08   # 0.08 %
_DEFAULT_TAKER_FEE = 0.10


def _to_okx_symbol(symbol: str) -> str:
    """'BTC/USDT' → 'BTC-USDT'"""
    return symbol.replace("/", "-")


class OKXConnector(BaseExchange):
    """OKX spot market connector (V5 API)."""

    def __init__(self, api_key: str = "", api_secret: str = "",
                 sandbox: bool = False):
        super().__init__(api_key, api_secret, sandbox)
        self._name = "okx"
        self._base = _BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        session = await self._get_session()
        url = f"{self._base}{path}"
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if data.get("code", "0") != "0":
                    raise ValueError(
                        f"OKX error {data.get('code')}: {data.get('msg')}")
                return data.get("data", [])
        except Exception as exc:
            logger.warning("OKX API error (%s): %s – using simulated data",
                           path, exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        okx_sym = _to_okx_symbol(symbol)
        try:
            data = await self._get("/api/v5/market/ticker",
                                   {"instId": okx_sym})
            t = data[0]
            bid  = float(t.get("bidPx", 0))
            ask  = float(t.get("askPx", 0))
            last = float(t.get("last", 0))
            vol  = float(t.get("vol24h", 0))
            logger.debug("OKX ticker %s: bid=%.4f ask=%.4f", symbol, bid, ask)
            return Ticker(
                exchange=self.name, symbol=symbol,
                bid=bid, ask=ask, last=last, volume_24h=vol,
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("OKX get_ticker failed for %s – using simulation",
                           symbol)
            return self._simulated_ticker(symbol)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        okx_sym = _to_okx_symbol(symbol)
        sz = min(depth, 50)
        try:
            data = await self._get("/api/v5/market/books",
                                   {"instId": okx_sym, "sz": sz})
            ob = data[0]
            bids = [[float(p), float(v)] for p, v, *_ in ob.get("bids", [])]
            asks = [[float(p), float(v)] for p, v, *_ in ob.get("asks", [])]
            return OrderBook(
                exchange=self.name, symbol=symbol,
                bids=sorted(bids, key=lambda x: -x[0]),
                asks=sorted(asks, key=lambda x:  x[0]),
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("OKX get_orderbook failed for %s – using simulation",
                           symbol)
            return self._simulated_orderbook(symbol)

    async def get_trading_fees(self, symbol: str) -> FeeSchedule:
        return FeeSchedule(
            exchange=self.name, symbol=symbol,
            maker_fee_pct=_DEFAULT_MAKER_FEE,
            taker_fee_pct=_DEFAULT_TAKER_FEE,
            withdrawal_fee_flat=0.0004,
            withdrawal_fee_pct=0.0,
        )

    async def get_symbol_list(self) -> list[TradingPair]:
        try:
            data = await self._get("/api/v5/public/instruments",
                                   {"instType": "SPOT"})
            pairs = []
            for item in data:
                if item.get("state") != "live":
                    continue
                base  = item.get("baseCcy", "")
                quote = item.get("quoteCcy", "")
                pairs.append(TradingPair(
                    exchange=self.name,
                    symbol=f"{base}/{quote}",
                    base_asset=base, quote_asset=quote,
                    min_order_size=float(item.get("minSz", 0)),
                    max_order_size=float(item.get("maxSz", 1e9) or 1e9),
                    price_precision=int(item.get("tickSz", "0.01").count("0")),
                    quantity_precision=int(item.get("lotSz", "0.0001").count("0")),
                ))
            return pairs
        except Exception:
            return self._simulated_symbol_list()

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _simulated_ticker(self, symbol: str) -> Ticker:
        import random
        base_prices = {
            "BTC/USDT": 65_050, "ETH/USDT": 3_498,
            "SOL/USDT": 150,    "BNB/USDT": 580,
            "XRP/USDT": 0.61,   "DOGE/USDT": 0.119,
        }
        base = base_prices.get(symbol, 100)
        base  *= random.uniform(0.9993, 1.0007)
        spread = base * 0.00016
        return Ticker(
            exchange=self.name, symbol=symbol,
            bid=round(base - spread / 2, 4),
            ask=round(base + spread / 2, 4),
            last=round(base + random.uniform(-spread, spread), 4),
            volume_24h=round(random.uniform(3000, 100_000), 2),
            timestamp=time.time(),
        )

    def _simulated_orderbook(self, symbol: str) -> OrderBook:
        import random
        ticker = self._simulated_ticker(symbol)
        mid = ticker.mid
        bids, asks = [], []
        for i in range(20):
            bp = mid * (1 - 0.00010 * (i + 1) - random.uniform(0, 0.00005))
            bv = random.uniform(0.1, 8.0)
            bids.append([round(bp, 4), round(bv, 4)])
        for i in range(20):
            ap = mid * (1 + 0.00010 * (i + 1) + random.uniform(0, 0.00005))
            av = random.uniform(0.1, 8.0)
            asks.append([round(ap, 4), round(av, 4)])
        return OrderBook(
            exchange=self.name, symbol=symbol,
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x:  x[0]),
            timestamp=time.time(),
        )

    def _simulated_symbol_list(self) -> list[TradingPair]:
        symbols = [
            ("BTC", "USDT"), ("ETH", "USDT"),
            ("SOL", "USDT"), ("XRP", "USDT"),
        ]
        return [
            TradingPair(
                exchange=self.name,
                symbol=f"{b}/{q}", base_asset=b, quote_asset=q,
                min_order_size=0.000001, max_order_size=1e9,
                price_precision=4, quantity_precision=4,
            )
            for b, q in symbols
        ]
