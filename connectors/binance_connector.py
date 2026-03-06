"""
Binance Exchange Connector
==========================
Implements BaseExchange for the Binance spot API.
Uses aiohttp for async HTTP.  Falls back to realistic simulated data
if the network is unavailable (demo / test mode).
"""

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from connectors.base_exchange import (
    BaseExchange, FeeSchedule, OrderBook, Ticker, TradingPair,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.binance.com"
_BASE_URL_TESTNET = "https://testnet.binance.vision"

# Default Binance spot fees (public tier)
_DEFAULT_MAKER_FEE = 0.10   # 0.10 %
_DEFAULT_TAKER_FEE = 0.10


class BinanceConnector(BaseExchange):
    """Binance spot market connector."""

    def __init__(self, api_key: str = "", api_secret: str = "",
                 sandbox: bool = False):
        super().__init__(api_key, api_secret, sandbox)
        self._name = "binance"
        self._base = _BASE_URL_TESTNET if sandbox else _BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        session = await self._get_session()
        url = f"{self._base}{path}"
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            logger.warning("Binance API error (%s): %s – using simulated data",
                           path, exc)
            raise

    # ------------------------------------------------------------------
    # Public API implementations
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        binance_symbol = symbol.replace("/", "")
        try:
            data = await self._get("/api/v3/ticker/bookTicker",
                                   {"symbol": binance_symbol})
            stats = await self._get("/api/v3/ticker/24hr",
                                    {"symbol": binance_symbol})
            return Ticker(
                exchange=self.name,
                symbol=symbol,
                bid=float(data["bidPrice"]),
                ask=float(data["askPrice"]),
                last=float(stats["lastPrice"]),
                volume_24h=float(stats["volume"]),
                timestamp=time.time(),
            )
        except Exception:
            return self._simulated_ticker(symbol)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        binance_symbol = symbol.replace("/", "")
        # Binance valid limits: 5,10,20,50,100,500,1000
        limit = min(depth, 100)
        try:
            data = await self._get("/api/v3/depth",
                                   {"symbol": binance_symbol, "limit": limit})
            return OrderBook(
                exchange=self.name,
                symbol=symbol,
                bids=[[float(p), float(v)] for p, v in data["bids"]],
                asks=[[float(p), float(v)] for p, v in data["asks"]],
                timestamp=time.time(),
            )
        except Exception:
            return self._simulated_orderbook(symbol)

    async def get_trading_fees(self, symbol: str) -> FeeSchedule:
        # Authenticated endpoint – skip if no key; return public defaults
        return FeeSchedule(
            exchange=self.name,
            symbol=symbol,
            maker_fee_pct=_DEFAULT_MAKER_FEE,
            taker_fee_pct=_DEFAULT_TAKER_FEE,
            withdrawal_fee_flat=0.0005,   # BTC example
            withdrawal_fee_pct=0.0,
        )

    async def get_symbol_list(self) -> list[TradingPair]:
        try:
            data = await self._get("/api/v3/exchangeInfo")
            pairs = []
            for s in data["symbols"]:
                if s["status"] != "TRADING":
                    continue
                filters = {f["filterType"]: f for f in s["filters"]}
                lot = filters.get("LOT_SIZE", {})
                price_f = filters.get("PRICE_FILTER", {})
                pairs.append(TradingPair(
                    exchange=self.name,
                    symbol=f"{s['baseAsset']}/{s['quoteAsset']}",
                    base_asset=s["baseAsset"],
                    quote_asset=s["quoteAsset"],
                    min_order_size=float(lot.get("minQty", 0)),
                    max_order_size=float(lot.get("maxQty", 1e9)),
                    price_precision=int(s.get("quotePrecision", 8)),
                    quantity_precision=int(s.get("baseAssetPrecision", 8)),
                ))
            return pairs
        except Exception:
            return self._simulated_symbol_list()

    # ------------------------------------------------------------------
    # Simulation helpers (used when live API is unavailable)
    # ------------------------------------------------------------------

    def _simulated_ticker(self, symbol: str) -> Ticker:
        import random
        base_prices = {"BTC/USDT": 65_000, "ETH/USDT": 3_500,
                       "SOL/USDT": 150, "BNB/USDT": 580}
        base = base_prices.get(symbol, 100)
        spread = base * 0.0002
        return Ticker(
            exchange=self.name, symbol=symbol,
            bid=round(base - spread / 2, 2),
            ask=round(base + spread / 2, 2),
            last=round(base + random.uniform(-spread, spread), 2),
            volume_24h=round(random.uniform(1000, 50000), 2),
            timestamp=time.time(),
        )

    def _simulated_orderbook(self, symbol: str) -> OrderBook:
        import random
        ticker = self._simulated_ticker(symbol)
        mid = ticker.mid
        bids, asks = [], []
        for i in range(20):
            bp = mid * (1 - 0.0001 * (i + 1) - random.uniform(0, 0.00005))
            bv = random.uniform(0.1, 5.0)
            bids.append([round(bp, 2), round(bv, 4)])
        for i in range(20):
            ap = mid * (1 + 0.0001 * (i + 1) + random.uniform(0, 0.00005))
            av = random.uniform(0.1, 5.0)
            asks.append([round(ap, 2), round(av, 4)])
        return OrderBook(
            exchange=self.name, symbol=symbol,
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x: x[0]),
            timestamp=time.time(),
        )

    def _simulated_symbol_list(self) -> list[TradingPair]:
        symbols = [("BTC", "USDT"), ("ETH", "USDT"),
                   ("SOL", "USDT"), ("BNB", "USDT")]
        return [
            TradingPair(
                exchange=self.name,
                symbol=f"{b}/{q}", base_asset=b, quote_asset=q,
                min_order_size=0.00001, max_order_size=9000.0,
                price_precision=2, quantity_precision=5,
            )
            for b, q in symbols
        ]
