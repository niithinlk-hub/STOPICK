# STOCKER

STOCKER is a **Next.js (App Router, TypeScript, Tailwind)** technical stock-setup screener for NSE and
US equities. It is a ground-up port of the Python/Streamlit **STOPICK** workstation — the same
explainable, deterministic, rule-based signal engine — rebuilt as a modern, dark-mode trading web app
with an accuracy-tuned scoring model.

> Research tool, not investment advice. The confidence score is **not** a win-rate claim.

## What it does

- Scans bundled 250-name NSE + US universes, a 15-name quick sample, or a manual watchlist.
- Scores **breakout** and **pullback** setups from `0`–`100` with an A+/A/B/C/Reject grade.
- Blends trend alignment, market structure, breakout/pullback quality, volume, **momentum (RSI/MACD)**,
  **liquidity**, relative strength, volatility regime, **multi-timeframe agreement**,
  **follow-through (false-breakout guard)**, headroom, R:R, market regime, event risk and index alignment.
- Produces an **execution plan**: entry, structure + ATR stop, 1R/2R/3R targets, fixed-risk position size.
- Annotated **candlestick charts** (breakout line, retest/FVG zones, entry/stop/targets).
- **Relative-strength leaderboard**, **market-regime** dashboard, and a walk-forward + Monte Carlo
  **backtest**.

## Accuracy tuning vs STOPICK

STOCKER keeps STOPICK's deterministic, explainable core and adds confirmation factors that reduce false
positives:

| Addition | Why |
|----------|-----|
| **Momentum** (RSI zone quality + MACD histogram) | STOPICK computed RSI but never used it; rewards healthy-not-overbought trend momentum. |
| **Follow-through guard** | Penalizes upper-wick rejections / weak closes above the breakout — the classic false break. |
| **Over-extension penalty** | Distance above the 20-EMA in ATR units; fades stretched, mean-reversion-prone entries. |
| **Liquidity gate** | 20-bar turnover floor; illiquid names give unreliable, unexecutable signals. |
| **Multi-timeframe agreement** | Explicit factor rewarding stacks that are both aligned and strong. |
| **Sector relative-strength routing** | Compares to a sector benchmark when known (README TODO #1 in STOPICK). |
| **Anchored-VWAP & accumulation** | Above breakout-anchored VWAP + OBV slope feed the volume score. |

Weights live in `lib/config.ts` (`scoringProfiles`).

## Stack

- Next.js 14 (App Router) · React 18 · TypeScript (strict)
- Tailwind CSS (semantic design tokens, dark-first) · framer-motion
- `lightweight-charts` (candlesticks) · `recharts` (analytics) · `lucide-react` (icons)
- `yahoo-finance2` (server-side OHLCV) · `zod` (API validation)

## Getting started

```bash
cd stocker
npm install
npm run dev
# open http://localhost:3000
```

Type-check / build:

```bash
npm run typecheck
npm run build
```

## Architecture

```
stocker/
├── app/
│   ├── layout.tsx                # fonts + AppShell
│   ├── page.tsx                  # Scanner
│   ├── top-setups/ relative-strength/ regime/ backtest/
│   └── api/
│       ├── scan/route.ts         # POST → runScan
│       ├── symbol/[ticker]/route.ts
│       └── backtest/[ticker]/route.ts
├── lib/
│   ├── config.ts                 # scoring profiles, benchmark + sector maps, runtime
│   ├── cn.ts · format.ts · grades.ts
│   ├── client/api.ts             # typed client fetchers
│   ├── data/                     # yahoo provider, TTL cache, universes
│   └── engine/                   # types, indicators, trend, structure, breakout,
│                                 # pullback, momentum, liquidity, context, scoring,
│                                 # risk, backtest, scan (orchestrator)
└── components/
    ├── AppShell.tsx · ui/primitives.tsx
    ├── scanner/ top/ rs/ regime/ backtest/
```

The engine is pure, server-only TypeScript. Each module mirrors a STOPICK Python module
(`signals/`, `scoring/`, `risk/`, `backtest/`) so the logic stays auditable.

## Data & limits

- OHLCV comes from Yahoo Finance via `yahoo-finance2` (server routes only), cached in-process for 30 min.
- Intraday history is limited by Yahoo (15m ≈ 60 days, 60m ≈ 2 years); 4h bars are resampled from 60m.
- NSE tickers are normalized with `.NS`; benchmarks are `^NSEI` (NSE) and `SPY` (US).
- Large scans (250 names) take time and are subject to Yahoo rate limits — start with the quick sample.
- Confirmed pivots use left/right bars, so structure signals are intentionally delayed (no repainting).
