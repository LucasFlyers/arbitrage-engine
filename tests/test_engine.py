"""
Test Suite – Arbitrage & Risk Monitoring Engine
===============================================
Covers: spread engine, slippage simulator, risk model, opportunity engine,
        cache layer, and exchange connectors (simulated mode).

Run:  pytest tests/ -v
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.base_exchange import FeeSchedule, OrderBook, Ticker
from connectors.binance_connector import BinanceConnector
from connectors.kraken_connector import KrakenConnector
from core.spread_engine import SlippageSimulator, SpreadEngine
from core.risk_model import RiskModel
from core.opportunity_engine import OpportunityEngine
from data.market_cache import MarketCache, MarketDataCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ticker(exchange: str, bid: float, ask: float,
                symbol: str = "BTC/USDT") -> Ticker:
    return Ticker(
        exchange=exchange, symbol=symbol,
        bid=bid, ask=ask, last=(bid + ask) / 2,
        volume_24h=10_000.0, timestamp=time.time(),
    )


def make_orderbook(exchange: str, mid: float,
                   symbol: str = "BTC/USDT",
                   spread_pct: float = 0.02,
                   depth: int = 10,
                   bid_volume: float = 1.0,
                   ask_volume: float = 1.0) -> OrderBook:
    half = mid * spread_pct / 200
    bids = [[round(mid - half - i * mid * 0.0001, 2),
             round(bid_volume, 4)] for i in range(depth)]
    asks = [[round(mid + half + i * mid * 0.0001, 2),
             round(ask_volume, 4)] for i in range(depth)]
    return OrderBook(
        exchange=exchange, symbol=symbol,
        bids=sorted(bids, key=lambda x: -x[0]),
        asks=sorted(asks, key=lambda x: x[0]),
        timestamp=time.time(),
    )


def make_fees(exchange: str, maker: float = 0.10,
              taker: float = 0.10) -> FeeSchedule:
    return FeeSchedule(
        exchange=exchange, symbol="BTC/USDT",
        maker_fee_pct=maker, taker_fee_pct=taker,
        withdrawal_fee_flat=0.0005, withdrawal_fee_pct=0.0,
    )


# ---------------------------------------------------------------------------
# Ticker & OrderBook sanity
# ---------------------------------------------------------------------------

class TestDataModels:
    def test_ticker_mid(self):
        t = make_ticker("binance", bid=65_000, ask=65_010)
        assert t.mid == pytest.approx(65_005.0)

    def test_orderbook_best_prices(self):
        ob = make_orderbook("binance", mid=65_000)
        assert ob.best_bid < 65_000
        assert ob.best_ask > 65_000
        assert ob.best_bid < ob.best_ask

    def test_orderbook_sorted(self):
        ob = make_orderbook("binance", mid=65_000)
        bids_prices = [b[0] for b in ob.bids]
        asks_prices = [a[0] for a in ob.asks]
        assert bids_prices == sorted(bids_prices, reverse=True)
        assert asks_prices == sorted(asks_prices)


# ---------------------------------------------------------------------------
# Slippage Simulator
# ---------------------------------------------------------------------------

class TestSlippageSimulator:
    def setup_method(self):
        self.sim = SlippageSimulator()

    def test_buy_small_no_slippage(self):
        """A tiny buy on a deep book should fill at best ask."""
        ob = make_orderbook("binance", mid=65_000, ask_volume=10.0)
        vwap = self.sim.simulate_buy(ob, notional_usd=100)
        best_ask = ob.best_ask
        assert vwap == pytest.approx(best_ask, rel=1e-3)

    def test_buy_large_causes_slippage(self):
        """A large buy on a thin book should produce positive slippage."""
        ob = make_orderbook("binance", mid=65_000,
                            ask_volume=0.01, depth=5)
        slippage = self.sim.estimate_slippage_pct(ob, notional_usd=100_000,
                                                  is_buy=True)
        assert slippage > 0

    def test_sell_small_no_slippage(self):
        ob = make_orderbook("binance", mid=65_000, bid_volume=10.0)
        vwap = self.sim.simulate_sell(ob, notional_usd=100)
        best_bid = ob.best_bid
        assert vwap == pytest.approx(best_bid, rel=1e-3)

    def test_slippage_pct_non_negative(self):
        ob = make_orderbook("binance", mid=65_000)
        for notional in [500, 5_000, 50_000]:
            s = self.sim.estimate_slippage_pct(ob, notional, is_buy=True)
            assert s >= 0


# ---------------------------------------------------------------------------
# Spread Engine
# ---------------------------------------------------------------------------

class TestSpreadEngine:
    def setup_method(self):
        self.engine = SpreadEngine(trade_size_usd=5_000)

    def _calc(self, buy_price: float, sell_price: float,
              buy_fee: float = 0.10, sell_fee: float = 0.10) -> object:
        buy_ticker = make_ticker("binance", bid=buy_price * 0.9999,
                                 ask=buy_price)
        sell_ticker = make_ticker("kraken", bid=sell_price,
                                  ask=sell_price * 1.0001)
        buy_ob = make_orderbook("binance", mid=buy_price)
        sell_ob = make_orderbook("kraken", mid=sell_price)
        buy_fees = make_fees("binance", taker=buy_fee)
        sell_fees = make_fees("kraken", taker=sell_fee)
        return self.engine.calculate(
            buy_ticker, buy_ob, buy_fees,
            sell_ticker, sell_ob, sell_fees,
        )

    def test_positive_raw_spread(self):
        result = self._calc(65_000, 65_200)
        assert result.raw_spread > 0
        assert result.spread_pct > 0

    def test_fee_reduces_spread(self):
        result = self._calc(65_000, 65_200)
        assert result.fee_adjusted_spread < result.raw_spread

    def test_slippage_further_reduces_spread(self):
        result = self._calc(65_000, 65_200)
        assert result.slippage_adjusted_spread <= result.fee_adjusted_spread

    def test_no_spread_not_profitable(self):
        result = self._calc(65_200, 65_000)   # inverted prices
        assert result.raw_spread < 0
        assert not result.is_profitable

    def test_to_dict_keys(self):
        result = self._calc(65_000, 65_200)
        d = result.to_dict()
        for key in ["raw_spread", "spread_pct", "fee_adjusted_spread",
                    "slippage_adjusted_spread", "is_profitable"]:
            assert key in d


# ---------------------------------------------------------------------------
# Risk Model
# ---------------------------------------------------------------------------

class TestRiskModel:
    def setup_method(self):
        self.model = RiskModel(depth_band_pct=0.5, trade_size_usd=5_000)

    def test_score_range(self):
        ob = make_orderbook("binance", mid=65_000)
        ticker = make_ticker("binance", bid=64_995, ask=65_005)
        score = self.model.evaluate(ob, ticker)
        for attr in ["liquidity_score", "imbalance_score",
                     "sustainability_score", "overall_risk_score"]:
            val = getattr(score, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of range"

    def test_thin_book_higher_risk(self):
        deep_ob = make_orderbook("binance", mid=65_000, ask_volume=50.0,
                                 bid_volume=50.0, depth=20)
        thin_ob = make_orderbook("binance", mid=65_000, ask_volume=0.01,
                                 bid_volume=0.01, depth=5)
        ticker = make_ticker("binance", bid=64_995, ask=65_005)
        deep_score = self.model.evaluate(deep_ob, ticker)
        thin_score = self.model.evaluate(thin_ob, ticker)
        assert thin_score.liquidity_score > deep_score.liquidity_score

    def test_imbalanced_book_higher_risk(self):
        balanced_ob = make_orderbook("binance", mid=65_000,
                                     bid_volume=5.0, ask_volume=5.0)
        skewed_ob = make_orderbook("binance", mid=65_000,
                                   bid_volume=50.0, ask_volume=0.5)
        ticker = make_ticker("binance", bid=64_995, ask=65_005)
        # Need fresh model instances so price history is clean
        m1 = RiskModel()
        m2 = RiskModel()
        balanced_score = m1.evaluate(balanced_ob, ticker)
        skewed_score = m2.evaluate(skewed_ob, ticker)
        assert skewed_score.imbalance_score > balanced_score.imbalance_score

    def test_label_mapping(self):
        ob = make_orderbook("binance", mid=65_000)
        ticker = make_ticker("binance", bid=64_995, ask=65_005)
        score = self.model.evaluate(ob, ticker)
        assert score.label in ["LOW", "MEDIUM", "HIGH", "EXTREME"]

    def test_volatility_increases_with_price_swings(self):
        model = RiskModel(price_history_len=10)
        ticker_base = make_ticker("binance", bid=64_995, ask=65_005)
        ob = make_orderbook("binance", mid=65_000)

        # Feed stable prices → low volatility
        for price in [65_000] * 10:
            t = make_ticker("binance", bid=price - 5, ask=price + 5)
            model.evaluate(ob, t)
        stable_score = model._volatility_score("binance", "BTC/USDT")

        # Feed wildly varying prices
        model2 = RiskModel(price_history_len=10)
        prices = [65_000, 66_000, 64_000, 67_000, 63_000,
                  68_000, 62_000, 69_000, 61_000, 70_000]
        for price in prices:
            t = make_ticker("binance", bid=price - 5, ask=price + 5)
            ob2 = make_orderbook("binance", mid=price)
            model2.evaluate(ob2, t)
        volatile_score = model2._volatility_score("binance", "BTC/USDT")

        assert volatile_score > stable_score


# ---------------------------------------------------------------------------
# Opportunity Engine
# ---------------------------------------------------------------------------

class TestOpportunityEngine:
    def setup_method(self):
        self.engine = OpportunityEngine(
            min_spread_pct=0.05,
            target_spread_pct=0.30,
        )
        self._spread_engine = SpreadEngine(trade_size_usd=5_000)
        self._risk_model = RiskModel()

    def _make_opportunity(self, buy_price: float, sell_price: float):
        buy_ticker = make_ticker("binance", bid=buy_price * 0.9999,
                                 ask=buy_price)
        sell_ticker = make_ticker("kraken", bid=sell_price,
                                  ask=sell_price * 1.0001)
        buy_ob = make_orderbook("binance", mid=buy_price)
        sell_ob = make_orderbook("kraken", mid=sell_price)
        fees = make_fees("binance")
        spread = self._spread_engine.calculate(
            buy_ticker, buy_ob, fees,
            sell_ticker, sell_ob, make_fees("kraken"),
        )
        buy_risk = self._risk_model.evaluate(buy_ob, buy_ticker)
        sell_risk = self._risk_model.evaluate(sell_ob, sell_ticker)
        return self.engine.evaluate(spread, buy_risk, sell_risk)

    def test_tier_none_for_zero_spread(self):
        opp = self._make_opportunity(65_000, 65_000)
        assert opp.tier == "NONE"

    def test_execution_probability_range(self):
        opp = self._make_opportunity(65_000, 65_300)
        assert 0.0 <= opp.execution_probability <= 1.0

    def test_opportunity_score_range(self):
        opp = self._make_opportunity(65_000, 65_300)
        assert 0.0 <= opp.opportunity_score <= 100.0

    def test_large_spread_better_than_small(self):
        small_opp = self._make_opportunity(65_000, 65_050)
        large_opp = self._make_opportunity(65_000, 65_500)
        assert large_opp.execution_probability >= small_opp.execution_probability

    def test_to_dict_structure(self):
        opp = self._make_opportunity(65_000, 65_300)
        d = opp.to_dict()
        assert "spreads" in d
        assert "risk" in d
        assert "scoring" in d
        assert "tier" in d["scoring"]


# ---------------------------------------------------------------------------
# Market Cache
# ---------------------------------------------------------------------------

class TestMarketCache:
    def test_set_and_get(self):
        cache = MarketCache(default_ttl=10)
        asyncio.run(cache.set("key1", "value1"))
        val = asyncio.run(cache.get("key1"))
        assert val == "value1"

    def test_expired_returns_none(self):
        cache = MarketCache(default_ttl=0.01)
        asyncio.run(cache.set("key2", "value2"))
        time.sleep(0.05)
        val = asyncio.run(cache.get("key2"))
        assert val is None

    def test_miss_returns_none(self):
        cache = MarketCache()
        val = asyncio.run(cache.get("does_not_exist"))
        assert val is None

    def test_stats(self):
        cache = MarketCache(default_ttl=10)
        asyncio.run(cache.set("k", "v"))
        asyncio.run(cache.get("k"))      # hit
        asyncio.run(cache.get("nope"))   # miss
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

    def test_purge_expired(self):
        cache = MarketCache(default_ttl=0.01)
        asyncio.run(cache.set("a", 1))
        asyncio.run(cache.set("b", 2))
        time.sleep(0.05)
        purged = asyncio.run(cache.purge_expired())
        assert purged == 2


# ---------------------------------------------------------------------------
# Exchange Connectors (simulated)
# ---------------------------------------------------------------------------

class TestBinanceConnector:
    def test_simulated_ticker(self):
        conn = BinanceConnector()
        ticker = conn._simulated_ticker("BTC/USDT")
        assert ticker.exchange == "binance"
        assert ticker.bid > 0
        assert ticker.ask > ticker.bid

    def test_simulated_orderbook(self):
        conn = BinanceConnector()
        ob = conn._simulated_orderbook("BTC/USDT")
        assert len(ob.bids) == 20
        assert len(ob.asks) == 20
        assert ob.best_bid < ob.best_ask

    def test_simulated_symbol_list(self):
        conn = BinanceConnector()
        symbols = conn._simulated_symbol_list()
        assert len(symbols) > 0
        assert all("/" in s.symbol for s in symbols)


class TestKrakenConnector:
    def test_simulated_ticker(self):
        conn = KrakenConnector()
        ticker = conn._simulated_ticker("BTC/USD")
        assert ticker.exchange == "kraken"
        assert ticker.ask > ticker.bid

    def test_normalise_asset(self):
        from connectors.kraken_connector import _normalise_asset
        assert _normalise_asset("XXBT") == "BTC"
        assert _normalise_asset("ZUSD") == "USD"
        assert _normalise_asset("SOL") == "SOL"   # passthrough
