# Arbitrage & Risk Monitoring Engine

A production-grade cryptocurrency arbitrage detection system built with clean
modular architecture, async I/O, and an AI-powered explanation layer.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     main.py  (ArbitrageEngine)                  │
│          orchestrates polling loop + component wiring           │
└─────────┬──────────┬─────────────┬────────────────┬────────────┘
          │          │             │                │
   ┌──────▼──┐  ┌────▼─────┐  ┌───▼──────┐  ┌─────▼──────┐
   │Exchange │  │ Spread   │  │  Risk    │  │Opportunity │
   │Connector│  │ Engine   │  │  Model   │  │  Engine    │
   │ Layer   │  │          │  │          │  │            │
   └──────┬──┘  └────┬─────┘  └───┬──────┘  └─────┬──────┘
          │          │             │                │
   ┌──────▼──┐  ┌────▼─────┐      │         ┌─────▼──────┐
   │ Market  │  │Slippage  │      │         │    AI      │
   │  Cache  │  │Simulator │      │         │ Explainer  │
   └─────────┘  └──────────┘      │         └────────────┘
                                  │
                          ┌───────▼──────┐
                          │Price History │
                          │ (Volatility) │
                          └──────────────┘
```

---

## Module Guide

### `connectors/`

| File | Purpose |
|------|---------|
| `base_exchange.py` | Abstract `BaseExchange` class + canonical data models (`Ticker`, `OrderBook`, `FeeSchedule`, `TradingPair`). All exchange connectors inherit from this. |
| `binance_connector.py` | Binance spot REST API connector. Auto-falls back to realistic simulated data when offline. |
| `kraken_connector.py` | Kraken REST API connector with asset normalisation (XXBT → BTC). Also falls back to simulation. |

### `core/`

| File | Purpose |
|------|---------|
| `spread_engine.py` | Calculates raw, fee-adjusted, and slippage-adjusted spreads. Contains the `SlippageSimulator` (VWAP book-walk). |
| `risk_model.py` | Four-dimensional risk scoring: liquidity depth, order book imbalance, volume sustainability, volatility. |
| `opportunity_engine.py` | Combines spread + risk into a scored `ArbitrageOpportunity` with execution probability and tier (NONE / MARGINAL / GOOD / EXCELLENT). |

### `ai/`

| File | Purpose |
|------|---------|
| `opportunity_explainer.py` | Calls Claude via the Anthropic API to produce natural-language market analysis. Includes a rule-based fallback for offline use. |

### `data/`

| File | Purpose |
|------|---------|
| `market_cache.py` | Async TTL key-value cache with per-data-type TTLs. Prevents redundant API calls within a polling cycle. |

### `config/`

| File | Purpose |
|------|---------|
| `settings.yaml` | All tunable parameters: exchange credentials, symbols, polling interval, fee thresholds, scoring weights, AI settings, logging. |

---

## Spread Calculation

```
raw_spread              = sell_bid  −  buy_ask

fee_cost                = buy_ask × (taker_fee_buy + taker_fee_sell + withdrawal_pct)

fee_adjusted_spread     = raw_spread − fee_cost

slippage_drag           = buy_ask × (buy_slippage_pct + sell_slippage_pct) / 100

slippage_adjusted_spread = fee_adjusted_spread − slippage_drag
```

**Slippage** is estimated by walking the live order book level-by-level and
computing the volume-weighted average price (VWAP) required to fill the full
notional trade size.

---

## Risk Scoring

Each sub-score is in **[0, 1]** (0 = low risk, 1 = high risk).

| Score | Formula |
|-------|---------|
| **Liquidity** | `1 / (1 + total_usd_within_band / 50_000)` – sigmoid on USD volume within 0.5% of mid |
| **Imbalance** | `min(1, |log(bid_vol / ask_vol)| / 2.3)` – log-scale asymmetry |
| **Sustainability** | `1 / (1 + available_usd_1pct_band / 2 / trade_size)` – how many times trade fits |
| **Volatility** | `min(1, stdev(prices) / mean(prices) / 0.005)` – coefficient of variation of recent mid prices |

**Overall** = weighted sum (configurable in `settings.yaml`).

---

## Opportunity Scoring

```
spread_score       = sigmoid(net_spread_pct, target=0.30%)
liquidity_score    = 1 − avg_liquidity_risk
sustainability     = 1 − avg_sustainability_risk
volatility_penalty = 1 − avg_volatility_risk

execution_prob = Σ weight_i × score_i     (sum-to-1 weights)

opportunity_score = exec_prob × 100 × (1 − combined_risk × 0.3)
```

**Tier classification:**

| Tier | Conditions |
|------|-----------|
| EXCELLENT | exec_prob ≥ 0.70  AND  net_spread ≥ 0.20% |
| GOOD | exec_prob ≥ 0.55  AND  net_spread ≥ 0.10% |
| MARGINAL | profitable but below GOOD thresholds |
| NONE | spread doesn't cover costs |

---

## Example Output

```
╔══════════════════════════════════════════════════╗
║  ARBITRAGE OPPORTUNITY DETECTED  [GOOD]
╠══════════════════════════════════════════════════╣
║  Symbol   : BTC/USDT                            ║
║  Route    : BINANCE      →  KRAKEN              ║
║  Raw Sprd : 0.3077%   Fee Adj: 0.1077%          ║
║  Net Sprd : 0.1077%   Risk   : 0.076            ║
║  Exec Prob: 62.10%    Score  : 57.6/100         ║
╚══════════════════════════════════════════════════╝

── AI ANALYSIS ──────────────────────────────────────
The spread between Binance and Kraken arises from a
temporary liquidity imbalance: Kraken's ask side is
thinner relative to Binance's, allowing a price
discrepancy to persist momentarily...

  → Action    : CONSIDER – Viable opportunity. Verify live book before trading.
  → Confidence: MEDIUM
────────────────────────────────────────────────────
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install aiohttp pyyaml
# For tests:
pip install pytest pytest-asyncio
```

### 2. Configure

Edit `config/settings.yaml` or use environment variables:

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export KRAKEN_API_KEY="your_key"
export KRAKEN_API_SECRET="your_secret"
export ANTHROPIC_API_KEY="your_key"   # for AI explanations
```

> **No API keys required** – the engine auto-uses realistic simulated market
> data when live APIs are unavailable.

### 3. Run

```bash
cd arbitrage_engine
python main.py

# Override symbols and poll interval:
python main.py --symbols BTC/USDT ETH/USDT --poll 10

# Disable AI explanations:
python main.py --no-ai
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Adding a New Exchange

1. Create `connectors/myexchange_connector.py`
2. Subclass `BaseExchange` and implement all 4 abstract methods
3. Add simulated fallback methods (`_simulated_ticker`, etc.)
4. Register in `main.py → ArbitrageEngine._init_exchanges()`
5. Add credentials block to `config/settings.yaml`

---

## Production Scaling Suggestions

| Concern | Recommendation |
|---------|---------------|
| Latency | Replace REST polling with WebSocket streams (Binance `bookTicker` stream, Kraken `ticker` feed) |
| Throughput | Run one process per symbol using `asyncio.TaskGroup` or distribute across workers via Redis pub/sub |
| Persistence | Pipe opportunity events to TimescaleDB or InfluxDB for backtesting |
| Alerting | Add Discord / Telegram webhook in `_log_opportunity()` |
| Execution | Connect to exchange order APIs in a separate `executor/` module with dry-run mode |
| Risk limits | Add position size limits, daily loss limits, and circuit breakers in `OpportunityEngine` |
| Secrets | Store API keys in HashiCorp Vault or AWS Secrets Manager rather than env vars |
| Monitoring | Expose Prometheus metrics (cycle latency, opportunity rate, cache hit rate) |
