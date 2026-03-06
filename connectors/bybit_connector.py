"""
Bybit Exchange Connector
========================
Implements BaseExchange for the Bybit V5 REST API (spot market).
Bybit is accessible in Nigeria and most regions without a VPN.

API docs: https://bybit-exchange.github.io/docs/v5/intro
Public endpoints require no API key.

Symbol format: Bybit uses BTCUSDT (no slash) internally.
This connector accepts the canonical BASE/QUOTE format (e.g. BTC/USDT)
and converts automatically.
"""

import logging
import time
from typing import Optional

import aiohttp

from connectors.base_exchange import (
    BaseExchange, FeeSchedule, OrderBook, Ticker, TradingPair,
)

logger = logging.getLogger(__name__)

_BASE_URL      = "https://api.bybit.com"
_BASE_URL_TEST = "https://api-testnet.bybit.com"

# Bybit spot fees (public / standard tier)
_DEFAULT_MAKER_FEE = 0.10   # 0.10 %
_DEFAULT_TAKER_FEE = 0.10


def _to_bybit_symbol(symbol: str) -> str:
    """'BTC/USDT' → 'BTCUSDT'"""
    return symbol.replace("/", "")


class BybitConnector(BaseExchange):
    """Bybit spot market connector (V5 API)."""

    def __init__(self, api_key: str = "", api_secret: str = "",
                 sandbox: bool = False):
        super().__init__(api_key, api_secret, sandbox)
        self._name = "bybit"
        self._base = _BASE_URL_TEST if sandbox else _BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def _get(self, path: str, params: dict | None = None) -> dict:
        session = await self._get_session()
        url = f"{self._base}{path}"
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                # Bybit returns retCode 0 on success
                if data.get("retCode", 0) != 0:
                    raise ValueError(
                        f"Bybit error {data.get('retCode')}: "
                        f"{data.get('retMsg')}"
                    )
                return data.get("result", {})
        except Exception as exc:
            logger.warning("Bybit API error (%s): %s – using simulated data",
                           path, exc)
            raise

    # ------------------------------------------------------------------
    # Public API implementations
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        bybit_sym = _to_bybit_symbol(symbol)
        try:
            result = await self._get(
                "/v5/market/tickers",
                {"category": "spot", "symbol": bybit_sym},
            )
            items = result.get("list", [])
            if not items:
                raise ValueError(f"No ticker data for {bybit_sym}")
            t = items[0]
            bid  = float(t.get("bid1Price", 0))
            ask  = float(t.get("ask1Price", 0))
            last = float(t.get("lastPrice", 0))
            vol  = float(t.get("volume24h", 0))
            logger.debug("Bybit ticker %s: bid=%.4f ask=%.4f", symbol, bid, ask)
            return Ticker(
                exchange=self.name, symbol=symbol,
                bid=bid, ask=ask, last=last, volume_24h=vol,
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("Bybit get_ticker failed for %s – using simulation",
                           symbol)
            return self._simulated_ticker(symbol)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        bybit_sym = _to_bybit_symbol(symbol)
        # Bybit valid limits: 1, 25, 50, 200 (spot)
        limit = 25 if depth <= 25 else 50
        try:
            result = await self._get(
                "/v5/market/orderbook",
                {"category": "spot", "symbol": bybit_sym, "limit": limit},
            )
            bids = [[float(p), float(v)] for p, v in result.get("b", [])]
            asks = [[float(p), float(v)] for p, v in result.get("a", [])]
            return OrderBook(
                exchange=self.name, symbol=symbol,
                bids=sorted(bids, key=lambda x: -x[0]),
                asks=sorted(asks, key=lambda x:  x[0]),
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("Bybit get_orderbook failed for %s – using simulation",
                           symbol)
            return self._simulated_orderbook(symbol)

    async def get_trading_fees(self, symbol: str) -> FeeSchedule:
        # Public tier fees; authenticated endpoint needed for personalised rates
        return FeeSchedule(
            exchange=self.name, symbol=symbol,
            maker_fee_pct=_DEFAULT_MAKER_FEE,
            taker_fee_pct=_DEFAULT_TAKER_FEE,
            withdrawal_fee_flat=0.0005,   # BTC example
            withdrawal_fee_pct=0.0,
        )

    async def get_symbol_list(self) -> list[TradingPair]:
        try:
            result = await self._get(
                "/v5/market/instruments-info",
                {"category": "spot"},
            )
            pairs = []
            for item in result.get("list", []):
                if item.get("status") != "Trading":
                    continue
                base  = item.get("baseCoin", "")
                quote = item.get("quoteCoin", "")
                lot   = item.get("lotSizeFilter", {})
                price = item.get("priceFilter", {})
                pairs.append(TradingPair(
                    exchange=self.name,
                    symbol=f"{base}/{quote}",
                    base_asset=base,
                    quote_asset=quote,
                    min_order_size=float(lot.get("minOrderQty", 0)),
                    max_order_size=float(lot.get("maxOrderQty", 1e9)),
                    price_precision=len(
                        price.get("tickSize", "0.01").rstrip("0").split(".")[-1]
                    ) if "." in price.get("tickSize", "1") else 0,
                    quantity_precision=len(
                        lot.get("basePrecision", "0.00001").rstrip("0").split(".")[-1]
                    ) if "." in lot.get("basePrecision", "1") else 0,
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
            "BTC/USDT": 65_100, "ETH/USDT": 3_505,
            "SOL/USDT": 151,    "BNB/USDT": 582,
            "XRP/USDT": 0.62,   "DOGE/USDT": 0.12,
        }
        base = base_prices.get(symbol, 100)
        base  *= random.uniform(0.9994, 1.0006)
        spread = base * 0.00018
        return Ticker(
            exchange=self.name, symbol=symbol,
            bid=round(base - spread / 2, 4),
            ask=round(base + spread / 2, 4),
            last=round(base + random.uniform(-spread, spread), 4),
            volume_24h=round(random.uniform(2000, 80_000), 2),
            timestamp=time.time(),
        )

    def _simulated_orderbook(self, symbol: str) -> OrderBook:
        import random
        ticker = self._simulated_ticker(symbol)
        mid = ticker.mid
        bids, asks = [], []
        for i in range(20):
            bp = mid * (1 - 0.00012 * (i + 1) - random.uniform(0, 0.00006))
            bv = random.uniform(0.1, 6.0)
            bids.append([round(bp, 4), round(bv, 4)])
        for i in range(20):
            ap = mid * (1 + 0.00012 * (i + 1) + random.uniform(0, 0.00006))
            av = random.uniform(0.1, 6.0)
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
            ("SOL", "USDT"), ("BNB", "USDT"),
            ("XRP", "USDT"), ("DOGE", "USDT"),
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
