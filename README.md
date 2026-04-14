# STOPICK

STOPICK is a modular Python + Streamlit trading workstation for identifying only high-quality technical setups across NSE and US equities. The design leans toward explainability, conservative signal quality, realistic execution planning, and backtesting without lookahead assumptions.

## What It Does

- Scans NSE and US sample/manual watchlists
- Scores breakout and pullback setups from `0` to `100`
- Combines trend, market structure, breakout quality, pullback quality, volume, regime, and relative strength
- Produces execution plans with entry, stop, 1R/2R/3R targets, and position sizing
- Includes walk-forward backtesting and Monte Carlo trade-sequence stress checks
- Provides a Streamlit workstation with pages for:
  - Scanner
  - Top setups
  - Market structure charts
  - Relative strength leaderboard
  - Backtest analytics
  - Trade journal
  - Watchlist
  - Alerts
  - Market regime dashboard

## Project Structure

```text
STOPICK/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── secrets.toml.example
├── stopick_app/
│   ├── alerts.py
│   └── workstation.py
├── backtest/
│   └── engine.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── config.yaml
│   ├── watchlist_nse_sample.txt
│   └── watchlist_us_sample.txt
├── data/
│   ├── __init__.py
│   ├── cache.py
│   ├── loaders.py
│   ├── providers.py
│   └── symbols.py
├── journal/
│   └── storage.py
├── risk/
│   └── planner.py
├── scoring/
│   └── engine.py
├── signals/
│   ├── __init__.py
│   ├── breakout.py
│   ├── common.py
│   ├── context.py
│   ├── models.py
│   ├── pullback.py
│   ├── structure.py
│   └── trend_alignment.py
├── tests/
│   ├── conftest.py
│   ├── test_backtest.py
│   ├── test_data.py
│   ├── test_scoring.py
│   └── test_signals.py
└── ui/
    └── dashboard.py
```

## Setup

1. Install Python `3.11` or `3.12`.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` if you want Telegram/email/webhook alerts.
5. Run the workstation:

```bash
streamlit run app.py
```

## Configuration

Core configuration lives in `config/config.yaml`.

Important areas:
- `runtime`: capital base, default minimum score, friction assumptions
- `data`: benchmark mapping and data-source defaults
- `scoring_profiles`: weight sets for breakout and pullback ranking

Secrets can go in:
- `.env`
- Streamlit secrets

Supported secret keys:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `EMAIL_FROM`
- `EMAIL_TO`
- `SMTP_HOST`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `WEBHOOK_URL`

## UI Workflow

### Scanner
- choose `NSE`, `US`, or `BOTH`
- choose sample or manual watchlist mode
- choose timeframe
- choose breakout/pullback/both
- set minimum score
- run scan

### Top Setups
- ranked shortlist of the best opportunities only

### Market Structure Charts
- annotated candlestick chart
- breakout line
- retest zone
- FVG zone
- PNG export when `kaleido` is available

### Relative Strength Leaderboard
- relative strength sorted independently from the blended setup score

### Backtest Analytics
- walk-forward style split
- simple slippage, brokerage, and tax friction
- Monte Carlo resampling of trade sequence

### Trade Journal
- persistent SQLite-backed notes per setup

### Alerts
- Telegram, email, and webhook hooks
- duplicate-prevention state file

## Engineering Notes

- Unified OHLCV schema:
  `datetime, open, high, low, close, volume, symbol`
- NSE tickers are normalized with `.NS`
- US tickers are used directly
- Benchmarks:
  - NSE: `^NSEI`
  - US: `SPY`
- Confirmed pivots use left/right bars to reduce repainting
- Breakout scoring is weighted and explainable
- Confidence score is not a win-rate claim

## Architecture Summary

### Data Engine
- pluggable provider abstraction
- default `yfinance`
- disk cache for repeated reads
- threaded symbol loading

### Trend Alignment Engine
- EMA stack
- price placement
- HH/HL style state
- ADX filter
- EMA slope

### Market Structure Engine
- confirmed pivots
- BOS / CHOCH
- equal highs/lows
- liquidity sweep
- order-block proxy
- FVG zone and mitigation

### Breakout Engine
- base breakout
- ATH breakout
- flag continuation
- cup-and-handle approximation
- volatility contraction breakout
- event continuation approximation

### Pullback Engine
- retest zone
- FVG fill
- reclaim / rejection logic
- lower-risk continuation entry planning

### Context Engine
- volume participation
- OBV confirmation
- VWAP alignment
- relative strength vs benchmark
- market regime summary

### Risk Engine
- fixed-risk position sizing
- structure stop
- ATR-aware stop fallback
- 1R / 2R / 3R target ladder

### Backtest Engine
- breakout-entry series with next-bar execution
- slippage, brokerage, taxes
- walk-forward style split
- Monte Carlo sequence analysis

## Streamlit Cloud Deployment

1. Push this folder to a GitHub repository named `STOPICK`.
2. In Streamlit Community Cloud create a new app from that repo.
3. Set the main file to `app.py`.
4. Copy values from `.streamlit/secrets.toml.example` into Streamlit secrets.
5. Deploy and rescan from the app UI.

## GitHub Setup

Suggested repository name:

```text
STOPICK
```

Suggested first commit message:

```text
Initial STOPICK trading workstation
```

## AI Extension Guide

The current score is deterministic and rule-based. A future AI ranking layer should:
- never replace the rule engine directly
- consume structured features from `SetupSignal`
- produce a secondary probability estimate
- log model version, feature window, and calibration stats
- keep the deterministic score visible for explainability

Recommended next AI steps:
1. Store historical setup outcomes with feature snapshots.
2. Train a calibrated classifier on out-of-sample data only.
3. Use AI as a ranking overlay, not as a hard entry trigger at first.
4. Track drift by market, sector, volatility regime, and earnings/results windows.

## Mock Layout Description

- Left sidebar for scan controls and filters
- Premium summary cards on top
- Main scanner table in the center
- Detail panel below with:
  - why-this-qualified explanation
  - confidence breakdown
  - execution plan
  - structure chart

## Testing

```bash
pytest
```

## Logic Assumptions

1. Confirmed pivots require right-side bars, so structure signals are slightly delayed by design.
2. Relative strength uses benchmark overlap and not synthetic calendar filling.
3. Event-risk proximity depends on available provider metadata.
4. Order blocks and cup-and-handle logic are heuristic approximations, not institutional truth labels.

## Overfitting Risks

1. Too many weights can produce fragile rankings if tuned on a narrow sample.
2. Breakout heuristics can become overly specific to a single bull-market regime.
3. Intraday parameter choices are especially vulnerable to regime drift.

## False Breakout Risks

1. Low-participation breakouts can fail quickly even with clean structure.
2. News-driven gaps can invalidate otherwise strong technical alignment.
3. Nearby higher-timeframe supply can cap follow-through.

## Best Validation Workflow

1. Start with daily data and the sample watchlists.
2. Validate the score distribution, not just the top few names.
3. Run walk-forward tests by market and by setup family.
4. Stress test trade ordering with Monte Carlo.
5. Review failed trades manually before changing weights.

## TODO Roadmap

1. Add sector ETF and sector-index relative strength routing.
2. Add earnings/results calendar providers beyond the default source.
3. Add multi-position portfolio simulation with correlation clusters.
4. Add more realistic intraday execution handling and opening-range setups.
5. Add AI-assisted probability ranking with calibration reporting.
