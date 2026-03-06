"""
Kraken Exchange Connector
=========================
Implements BaseExchange for the Kraken REST API v0.
Normalises Kraken's non-standard symbol names (XXBTZUSD → BTC/USD) back
to the canonical BASE/QUOTE format used everywhere in the engine.

USDT → USD translation
-----------------------
Kraken does not list USDT pairs (e.g. ETH/USDT). It uses USD instead.
This connector automatically translates any incoming USDT symbol to its
USD equivalent when querying Kraken, then returns the ticker/orderbook
tagged with the original USDT symbol so the rest of the engine stays
consistent. USD and USDT are treated as equivalent for spread purposes
(they trade within a few basis points of each other).
"""

import logging
import time
from typing import Optional

import aiohttp

from connectors.base_exchange import (
    BaseExchange, FeeSchedule, OrderBook, Ticker, TradingPair,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.kraken.com"

# Kraken uses non-standard prefixes for some assets
_ASSET_MAP = {
    "XXBT": "BTC", "XETH": "ETH", "XLTC": "LTC",
    "XXRP": "XRP", "XXLM": "XLM", "ZUSD": "USD",
    "ZEUR": "EUR", "ZGBP": "GBP", "XZEC": "ZEC",
    "XXLM": "XLM", "XICN": "ICN",
}

_DEFAULT_MAKER_FEE = 0.16
_DEFAULT_TAKER_FEE = 0.26


def _normalise_asset(raw: str) -> str:
    return _ASSET_MAP.get(raw, raw)


def _usdt_to_usd(symbol: str) -> str:
    """
    Translate a USDT-quoted symbol to its USD equivalent for Kraken.
    e.g.  'ETH/USDT' → 'ETH/USD'
          'BTC/USDT' → 'BTC/USD'
    Non-USDT symbols are returned unchanged.
    """
    if symbol.endswith("/USDT"):
        return symbol[:-4] + "USD"
    return symbol


class KrakenConnector(BaseExchange):
    """Kraken spot market connector."""

    def __init__(self, api_key: str = "", api_secret: str = "",
                 sandbox: bool = False):
        super().__init__(api_key, api_secret, sandbox)
        self._name = "kraken"
        self._session: Optional[aiohttp.ClientSession] = None
        # populated on first symbol fetch: canonical → kraken_pair_name
        self._pair_map: dict[str, str] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def _get(self, path: str, params: dict | None = None) -> dict:
        session = await self._get_session()
        url = f"{_BASE_URL}{path}"
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if data.get("error") and data["error"]:
                    raise ValueError(f"Kraken error: {data['error']}")
                return data["result"]
        except Exception as exc:
            logger.warning("Kraken API error (%s): %s – using simulated data",
                           path, exc)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_pair_map(self):
        """Build canonical→kraken_name map once on first use."""
        if self._pair_map:
            return
        try:
            result = await self._get("/0/public/AssetPairs")
            for kraken_name, info in result.items():
                if "." in kraken_name:          # skip .d (dark pool) variants
                    continue
                base = _normalise_asset(info.get("base", ""))
                quote = _normalise_asset(info.get("quote", ""))
                canonical = f"{base}/{quote}"
                self._pair_map[canonical] = kraken_name
            logger.info("Kraken pair map loaded: %d pairs", len(self._pair_map))
        except Exception:
            # Hard-coded fallback so the engine always has something to work with
            self._pair_map = {
                "BTC/USD":  "XXBTZUSD",
                "ETH/USD":  "XETHZUSD",
                "SOL/USD":  "SOLUSD",
                "BNB/USD":  "BNBUSD",
                "XRP/USD":  "XXRPZUSD",
            }
            logger.warning("Kraken pair map used fallback (%d pairs)",
                           len(self._pair_map))

    def _to_kraken(self, symbol: str) -> str:
        """
        Convert a canonical symbol to the Kraken pair name.
        Automatically tries the USD version if the USDT version isn't found.
        """
        # Direct lookup first
        if symbol in self._pair_map:
            return self._pair_map[symbol]
        # Try USDT → USD translation
        usd_symbol = _usdt_to_usd(symbol)
        if usd_symbol in self._pair_map:
            return self._pair_map[usd_symbol]
        # Last resort: strip slash
        return symbol.replace("/", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        await self._ensure_pair_map()
        kraken_sym = self._to_kraken(symbol)
        try:
            result = await self._get("/0/public/Ticker", {"pair": kraken_sym})
            key = list(result.keys())[0]
            t = result[key]
            bid  = float(t["b"][0])
            ask  = float(t["a"][0])
            last = float(t["c"][0])
            vol  = float(t["v"][1])     # rolling 24-h volume
            logger.debug("Kraken ticker %s (via %s): bid=%.4f ask=%.4f",
                         symbol, kraken_sym, bid, ask)
            return Ticker(
                exchange=self.name, symbol=symbol,
                bid=bid, ask=ask, last=last, volume_24h=vol,
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("Kraken get_ticker failed for %s – using simulation",
                           symbol)
            return self._simulated_ticker(symbol)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        await self._ensure_pair_map()
        kraken_sym = self._to_kraken(symbol)
        try:
            result = await self._get("/0/public/Depth",
                                     {"pair": kraken_sym, "count": depth})
            key = list(result.keys())[0]
            ob = result[key]
            bids = [[float(p), float(v)] for p, v, _ in ob["bids"]]
            asks = [[float(p), float(v)] for p, v, _ in ob["asks"]]
            return OrderBook(
                exchange=self.name, symbol=symbol,
                bids=sorted(bids, key=lambda x: -x[0]),
                asks=sorted(asks, key=lambda x: x[0]),
                timestamp=time.time(),
            )
        except Exception:
            logger.warning("Kraken get_orderbook failed for %s – using simulation",
                           symbol)
            return self._simulated_orderbook(symbol)

    async def get_trading_fees(self, symbol: str) -> FeeSchedule:
        return FeeSchedule(
            exchange=self.name, symbol=symbol,
            maker_fee_pct=_DEFAULT_MAKER_FEE,
            taker_fee_pct=_DEFAULT_TAKER_FEE,
            withdrawal_fee_flat=0.00015,
            withdrawal_fee_pct=0.0,
        )

    async def get_symbol_list(self) -> list[TradingPair]:
        try:
            result = await self._get("/0/public/AssetPairs")
            pairs = []
            for kraken_name, info in result.items():
                if "." in kraken_name:
                    continue
                base  = _normalise_asset(info.get("base", ""))
                quote = _normalise_asset(info.get("quote", ""))
                pairs.append(TradingPair(
                    exchange=self.name,
                    symbol=f"{base}/{quote}",
                    base_asset=base, quote_asset=quote,
                    min_order_size=float(info.get("ordermin", 0)),
                    max_order_size=1e9,
                    price_precision=int(info.get("pair_decimals", 5)),
                    quantity_precision=int(info.get("lot_decimals", 8)),
                ))
            return pairs
        except Exception:
            return self._simulated_symbol_list()

    # ------------------------------------------------------------------
    # Simulation helpers  (used only when live API is unreachable)
    # ------------------------------------------------------------------

    def _simulated_ticker(self, symbol: str) -> Ticker:
        import random
        # Realistic prices for common pairs – covers both USD and USDT labels
        base_prices = {
            "BTC/USD": 65_000, "BTC/USDT": 65_000,
            "ETH/USD":  3_500, "ETH/USDT":  3_500,
            "SOL/USD":    150, "SOL/USDT":    150,
            "BNB/USD":    580, "BNB/USDT":    580,
        }
        base = base_prices.get(symbol, base_prices.get(_usdt_to_usd(symbol), 100))
        base  *= random.uniform(0.9995, 1.0005)
        spread = base * 0.00025
        return Ticker(
            exchange=self.name, symbol=symbol,
            bid=round(base - spread / 2, 2),
            ask=round(base + spread / 2, 2),
            last=round(base + random.uniform(-spread, spread), 2),
            volume_24h=round(random.uniform(500, 20_000), 2),
            timestamp=time.time(),
        )

    def _simulated_orderbook(self, symbol: str) -> OrderBook:
        import random
        ticker = self._simulated_ticker(symbol)
        mid = ticker.mid
        bids, asks = [], []
        for i in range(20):
            bp = mid * (1 - 0.00015 * (i + 1) - random.uniform(0, 0.00008))
            bv = random.uniform(0.05, 3.0)
            bids.append([round(bp, 2), round(bv, 4)])
        for i in range(20):
            ap = mid * (1 + 0.00015 * (i + 1) + random.uniform(0, 0.00008))
            av = random.uniform(0.05, 3.0)
            asks.append([round(ap, 2), round(av, 4)])
        return OrderBook(
            exchange=self.name, symbol=symbol,
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x: x[0]),
            timestamp=time.time(),
        )

    def _simulated_symbol_list(self) -> list[TradingPair]:
        symbols = [("BTC", "USD"), ("ETH", "USD"),
                   ("SOL", "USD"), ("BNB", "USD")]
        return [
            TradingPair(
                exchange=self.name,
                symbol=f"{b}/{q}", base_asset=b, quote_asset=q,
                min_order_size=0.0001, max_order_size=1e9,
                price_precision=1, quantity_precision=8,
            )
            for b, q in symbols
        ]
