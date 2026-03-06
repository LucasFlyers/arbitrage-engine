"""
Gate.io Exchange Connector
==========================
Implements BaseExchange for the Gate.io REST API v4 (spot market).
Gate.io is accessible globally including Nigeria without a VPN.

API docs: https://www.gate.io/docs/developers/apiv4/
Public endpoints require no API key.

Symbol format: Gate.io uses BTC_USDT (underscore) internally.
"""

import logging
import time
from typing import Optional

import aiohttp

from connectors.base_exchange import (
    BaseExchange, FeeSchedule, OrderBook, Ticker, TradingPair,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.gateio.ws/api/v4"

_DEFAULT_MAKER_FEE = 0.10
_DEFAULT_TAKER_FEE = 0.10


def _to_gate_symbol(symbol: str) -> str:
    """'BTC/USDT' → 'BTC_USDT'"""
    return symbol.replace("/", "_")


class GateConnector(BaseExchange):
    """Gate.io spot market connector (API v4)."""

    def __init__(self, api_key: str = "", api_secret: str = "",
                 sandbox: bool = False):
        super().__init__(api_key, api_secret, sandbox)
        self._name = "gate"
        self._base = _BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def _get(self, path: str, params: dict | None = None):
        session = await self._get_session()
        url = f"{self._base}{path}"
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            logger.warning("Gate.io API error (%s): %s – using simulated data",
                           path, exc)
            raise

    async def get_ticker(self, symbol: str) -> Ticker:
        gate_sym = _to_gate_symbol(symbol)
        try:
            data = await self._get("/spot/tickers", {"currency_pair": gate_sym})
            t = data[0] if isinstance(data, list) else data
            last  = float(t.get("last", 0))
            bid   = float(t.get("highest_bid", 0))
            ask   = float(t.get("lowest_ask", 0))
            vol   = float(t.get("base_volume", 0))
            # If bid/ask missing, estimate from last
            if bid == 0: bid = last * 0.9999
            if ask == 0: ask = last * 1.0001
            logger.debug("Gate ticker %s: bid=%.4f ask=%.4f", symbol, bid, ask)
            return Ticker(
                exchange=self.name, symbol=symbol,
                bid=bid, ask=ask, last=last, volume_24h=vol,
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("Gate get_ticker failed for %s – using simulation", symbol)
            return self._simulated_ticker(symbol)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        gate_sym = _to_gate_symbol(symbol)
        try:
            data = await self._get("/spot/order_book",
                                   {"currency_pair": gate_sym, "limit": min(depth, 50)})
            bids = [[float(p), float(v)] for p, v in data.get("bids", [])]
            asks = [[float(p), float(v)] for p, v in data.get("asks", [])]
            return OrderBook(
                exchange=self.name, symbol=symbol,
                bids=sorted(bids, key=lambda x: -x[0]),
                asks=sorted(asks, key=lambda x:  x[0]),
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("Gate get_orderbook failed for %s – using simulation", symbol)
            return self._simulated_orderbook(symbol)

    async def get_trading_fees(self, symbol: str) -> FeeSchedule:
        return FeeSchedule(
            exchange=self.name, symbol=symbol,
            maker_fee_pct=_DEFAULT_MAKER_FEE,
            taker_fee_pct=_DEFAULT_TAKER_FEE,
            withdrawal_fee_flat=0.0005,
            withdrawal_fee_pct=0.0,
        )

    async def get_symbol_list(self) -> list[TradingPair]:
        try:
            data = await self._get("/spot/currency_pairs")
            pairs = []
            for item in data:
                if item.get("trade_status") != "tradable":
                    continue
                base  = item.get("base", "")
                quote = item.get("quote", "")
                pairs.append(TradingPair(
                    exchange=self.name,
                    symbol=f"{base}/{quote}",
                    base_asset=base, quote_asset=quote,
                    min_order_size=float(item.get("min_base_amount", 0) or 0),
                    max_order_size=float(item.get("max_base_amount", 1e9) or 1e9),
                    price_precision=int(item.get("precision", 4)),
                    quantity_precision=int(item.get("amount_precision", 4)),
                ))
            return pairs
        except Exception:
            return self._simulated_symbol_list()

    def _simulated_ticker(self, symbol: str) -> Ticker:
        import random
        base_prices = {
            "BTC/USDT": 64_900, "ETH/USDT": 3_480,
            "SOL/USDT": 148,    "XRP/USDT": 0.605,
        }
        base   = base_prices.get(symbol, 100) * random.uniform(0.9990, 1.0010)
        spread = base * 0.00020
        return Ticker(
            exchange=self.name, symbol=symbol,
            bid=round(base - spread / 2, 4),
            ask=round(base + spread / 2, 4),
            last=round(base, 4),
            volume_24h=round(random.uniform(1000, 50_000), 2),
            timestamp=time.time(),
        )

    def _simulated_orderbook(self, symbol: str) -> OrderBook:
        import random
        ticker = self._simulated_ticker(symbol)
        mid = ticker.mid
        bids = [[round(mid * (1 - 0.00012*(i+1)), 4), round(random.uniform(0.1, 5), 4)] for i in range(20)]
        asks = [[round(mid * (1 + 0.00012*(i+1)), 4), round(random.uniform(0.1, 5), 4)] for i in range(20)]
        return OrderBook(
            exchange=self.name, symbol=symbol,
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x:  x[0]),
            timestamp=time.time(),
        )

    def _simulated_symbol_list(self) -> list[TradingPair]:
        return [TradingPair(exchange=self.name, symbol=f"{b}/USDT",
                            base_asset=b, quote_asset="USDT",
                            min_order_size=0.000001, max_order_size=1e9,
                            price_precision=4, quantity_precision=4)
                for b in ["BTC", "ETH", "SOL", "XRP"]]
