"""
Base Exchange Interface
=======================
Abstract base class defining the standard interface that all exchange
connectors must implement. Ensures uniform data formats across the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


# ---------------------------------------------------------------------------
# Canonical Data Models
# ---------------------------------------------------------------------------

@dataclass
class Ticker:
    exchange: str
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: float = field(default_factory=time.time)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class OrderBook:
    exchange: str
    symbol: str
    bids: list[list[float]]   # [[price, volume], ...]  descending
    asks: list[list[float]]   # [[price, volume], ...]  ascending
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float("inf")

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2


@dataclass
class FeeSchedule:
    exchange: str
    symbol: str
    maker_fee_pct: float          # e.g. 0.1  (= 0.1%)
    taker_fee_pct: float          # e.g. 0.1
    withdrawal_fee_flat: float    # flat fee in base currency
    withdrawal_fee_pct: float     # percentage of withdrawal amount


@dataclass
class TradingPair:
    exchange: str
    symbol: str          # e.g. "BTC/USDT"
    base_asset: str      # e.g. "BTC"
    quote_asset: str     # e.g. "USDT"
    min_order_size: float
    max_order_size: float
    price_precision: int
    quantity_precision: int


# ---------------------------------------------------------------------------
# Abstract Exchange Interface
# ---------------------------------------------------------------------------

class BaseExchange(ABC):
    """
    All exchange connectors must implement this interface.
    Provides standardised access to market data regardless of exchange.
    """

    def __init__(self, api_key: str = "", api_secret: str = "",
                 sandbox: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self._name: str = "base"

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Required implementations
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Return the current best bid/ask and last trade for a symbol."""
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Return the current order book up to `depth` levels per side."""
        ...

    @abstractmethod
    async def get_trading_fees(self, symbol: str) -> FeeSchedule:
        """Return the maker/taker and withdrawal fee schedule."""
        ...

    @abstractmethod
    async def get_symbol_list(self) -> list[TradingPair]:
        """Return all tradeable symbols on this exchange."""
        ...

    # ------------------------------------------------------------------
    # Optional helper – subclasses may override
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Ping the exchange and return True if reachable."""
        try:
            await self.get_symbol_list()
            return True
        except Exception:
            return False
