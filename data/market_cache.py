"""
Market Cache
============
In-memory TTL cache for market data.  Prevents redundant API calls within
the same monitoring cycle and provides a single source of truth for the
latest snapshots.

Key design decisions:
  • Thread-safe via asyncio.Lock (all callers are async coroutines)
  • Configurable per-data-type TTLs
  • Expiry returns None so callers can decide whether to refetch
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache Entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    value: Any
    expires_at: float

    def is_alive(self) -> bool:
        return time.time() < self.expires_at


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------

class MarketCache:
    """
    Generic async TTL key-value store.

    Usage
    -----
    cache = MarketCache(default_ttl=5.0)
    await cache.set("binance:BTC/USDT:ticker", ticker_obj, ttl=3.0)
    ticker = await cache.get("binance:BTC/USDT:ticker")
    """

    def __init__(self, default_ttl: float = 5.0):
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry and entry.is_alive():
                self._hits += 1
                return entry.value
            self._misses += 1
            if entry:
                del self._store[key]    # clean up expired
            return None

    async def set(self, key: str, value: Any,
                  ttl: float | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + effective_ttl
        async with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def purge_expired(self) -> int:
        """Remove all expired entries; returns count removed."""
        now = time.time()
        async with self._lock:
            expired = [k for k, v in self._store.items()
                       if v.expires_at <= now]
            for k in expired:
                del self._store[k]
        return len(expired)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "entries": len(self._store),
        }


# ---------------------------------------------------------------------------
# Typed Market Data Cache (convenience wrapper)
# ---------------------------------------------------------------------------

class MarketDataCache:
    """
    Higher-level cache wrapper with named helpers for each data type.
    Uses separate TTLs per data type.
    """

    _TTL = {
        "ticker":   3.0,
        "orderbook": 2.0,
        "fees":     300.0,    # fees change rarely
        "symbols":  3600.0,   # symbol list changes very rarely
    }

    def __init__(self):
        self._cache = MarketCache(default_ttl=5.0)

    # ------------------------------------------------------------------
    # Ticker
    # ------------------------------------------------------------------

    async def get_ticker(self, exchange: str, symbol: str):
        return await self._cache.get(f"{exchange}:{symbol}:ticker")

    async def set_ticker(self, exchange: str, symbol: str, ticker) -> None:
        await self._cache.set(
            f"{exchange}:{symbol}:ticker", ticker, ttl=self._TTL["ticker"])

    # ------------------------------------------------------------------
    # Order Book
    # ------------------------------------------------------------------

    async def get_orderbook(self, exchange: str, symbol: str):
        return await self._cache.get(f"{exchange}:{symbol}:orderbook")

    async def set_orderbook(self, exchange: str, symbol: str, ob) -> None:
        await self._cache.set(
            f"{exchange}:{symbol}:orderbook", ob, ttl=self._TTL["orderbook"])

    # ------------------------------------------------------------------
    # Fees
    # ------------------------------------------------------------------

    async def get_fees(self, exchange: str, symbol: str):
        return await self._cache.get(f"{exchange}:{symbol}:fees")

    async def set_fees(self, exchange: str, symbol: str, fees) -> None:
        await self._cache.set(
            f"{exchange}:{symbol}:fees", fees, ttl=self._TTL["fees"])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return self._cache.stats

    async def purge_expired(self) -> int:
        return await self._cache.purge_expired()
