/**
 * "RSI Confluence" strategy screen — a distinct, directional setup finder based on
 * the classic three-indicator confluence:
 *   1. Stochastic (14,3,3) — entry trigger: turning up out of oversold (long) /
 *      down out of overbought (short).
 *   2. RSI(14) vs the 50 mid-line — trend filter: >50 = long-only, <50 = short-only.
 *   3. MACD(12,26,9) — momentum confirmation: MACD line crossing/holding above the
 *      signal line for longs ("blue above red"), below for shorts.
 *
 * Exit framing (from the strategy): stop at the nearest swing low/high, target at
 * 1.5R, with a max-stop cap (adapted per timeframe for equities BTST/intraday).
 *
 * This is SEPARATE from the main composite scanner — it is a single, explicit
 * trade plan, not the 16-factor confluence grade. Pure + deterministic.
 */
import type { Bar, ConfluenceRow, Market, Timeframe } from "./types";
import { bounded, closes, highs, last, lows, macd, round, rsi, stochastic } from "./indicators";

const OVERSOLD = 20;
const OVERBOUGHT = 80;
const SWING_LOOKBACK = 10;

export interface ConfluenceCtx {
  ticker: string;
  market: Market;
  timeframe: Timeframe;
  /** Max acceptable stop distance as a fraction of entry; wider stops are penalized. */
  maxStopPct?: number;
}

export function analyzeConfluence(bars: Bar[], ctx: ConfluenceCtx): ConfluenceRow | null {
  if (bars.length < 40) return null;
  const c = closes(bars);
  const price = last(c);
  if (!Number.isFinite(price)) return null;

  const { k: kSeries, d: dSeries } = stochastic(bars, 14, 3, 3);
  const k = last(kSeries, 50);
  const d = last(dSeries, 50);
  const kPrev = kSeries.length >= 2 ? kSeries[kSeries.length - 2] : k;
  const dPrev = dSeries.length >= 2 ? dSeries[dSeries.length - 2] : d;

  const rsiV = last(rsi(c, 14), 50);

  const { macd: mlSeries, signal: sgSeries, histogram: hgSeries } = macd(c);
  const macdLine = last(mlSeries, 0);
  const sigLine = last(sgSeries, 0);
  const hist = last(hgSeries, 0);
  const histPrev = hgSeries.length >= 2 ? hgSeries[hgSeries.length - 2] : 0;
  const linePrev = mlSeries.length >= 2 ? mlSeries[mlSeries.length - 2] : macdLine;
  const sigPrev = sgSeries.length >= 2 ? sgSeries[sgSeries.length - 2] : sigLine;

  // Did %K visit an extreme in the last few bars (the "hit oversold/overbought" leg)?
  const recentK = kSeries.slice(-4, -1).filter((v) => Number.isFinite(v));
  const cameFromOversold = recentK.some((v) => v <= 25) || k <= 25;
  const cameFromOverbought = recentK.some((v) => v >= 75) || k >= 75;

  const stochCrossUp = kPrev <= dPrev && k > d;
  const stochCrossDown = kPrev >= dPrev && k < d;
  const macdCrossUp = linePrev <= sigPrev && macdLine > sigLine;
  const macdCrossDown = linePrev >= sigPrev && macdLine < sigLine;

  const stochBullish = (stochCrossUp || (k > d && cameFromOversold)) && k < OVERBOUGHT;
  const macdBullish = macdLine > sigLine && (macdCrossUp || (hist > 0 && hist > histPrev));
  const longSignal = stochBullish && rsiV > 50 && macdBullish;

  const stochBearish = (stochCrossDown || (k < d && cameFromOverbought)) && k > OVERSOLD;
  const macdBearish = macdLine < sigLine && (macdCrossDown || (hist < 0 && hist < histPrev));
  const shortSignal = stochBearish && rsiV < 50 && macdBearish;

  const side: ConfluenceRow["side"] = longSignal ? "long" : shortSignal ? "short" : "none";

  const stochState = k >= OVERBOUGHT ? "overbought" : k <= OVERSOLD ? "oversold" : k > d ? "rising" : "falling";
  const dp = price > 100 ? 2 : 3;
  const maxStopPct = ctx.maxStopPct ?? 0.04;

  let entry: number | null = null;
  let stop: number | null = null;
  let target: number | null = null;
  let stopPct: number | null = null;
  const reasons: string[] = [];

  if (side === "long") {
    entry = price;
    stop = Math.min(...lows(bars).slice(-SWING_LOOKBACK));
    const risk = entry - stop;
    stopPct = entry > 0 ? risk / entry : null;
    target = entry + 1.5 * risk;
    reasons.push(
      `Stochastic turning up (%K ${round(k, 1)} > %D ${round(d, 1)}) out of the oversold zone.`,
      `RSI ${round(rsiV, 1)} above 50 — uptrend filter passes (long-only).`,
      `MACD line above signal${macdCrossUp ? " (fresh bullish cross)" : ""} — momentum confirms.`,
      `Stop at swing low ${round(stop, dp)}, 1.5R target ${round(target, dp)}.`,
    );
  } else if (side === "short") {
    entry = price;
    stop = Math.max(...highs(bars).slice(-SWING_LOOKBACK));
    const risk = stop - entry;
    stopPct = entry > 0 ? risk / entry : null;
    target = entry - 1.5 * risk;
    reasons.push(
      `Stochastic turning down (%K ${round(k, 1)} < %D ${round(d, 1)}) out of the overbought zone.`,
      `RSI ${round(rsiV, 1)} below 50 — downtrend filter passes (short-only).`,
      `MACD line below signal${macdCrossDown ? " (fresh bearish cross)" : ""} — momentum confirms.`,
      `Stop at swing high ${round(stop, dp)}, 1.5R target ${round(target, dp)}.`,
    );
  } else {
    // Explain the missing leg(s) so the row is still informative.
    if (!(rsiV > 50) && !(rsiV < 50)) reasons.push("RSI sitting on the 50 line — no trend bias.");
    else reasons.push(rsiV > 50 ? "RSI > 50 (up-bias) but Stochastic/MACD not aligned for a long." : "RSI < 50 (down-bias) but Stochastic/MACD not aligned for a short.");
  }

  // Confidence: alignment + signal strength. Only meaningful when a side fires.
  let confidence = 0;
  if (side !== "none") {
    const stochScore = 35 * (stochCrossUp || stochCrossDown ? 1 : 0.7);
    const rsiScore = 30 * bounded(Math.abs(rsiV - 50) / 20);
    const histStrength = bounded(Math.abs(hist) / Math.max(price * 0.004, 1e-9));
    const macdScore = 35 * bounded(histStrength * 0.7 + (macdCrossUp || macdCrossDown ? 0.3 : 0.15));
    confidence = round(stochScore + rsiScore + macdScore, 0);
    if (stopPct != null && stopPct > maxStopPct) {
      confidence = round(confidence * 0.7, 0);
      reasons.push(`Stop is wide (${round(stopPct * 100, 2)}% > ${round(maxStopPct * 100, 1)}% cap) — size down or skip.`);
    }
  }

  return {
    ticker: ctx.ticker,
    market: ctx.market,
    timeframe: ctx.timeframe,
    side,
    signal: side !== "none",
    confidence,
    price: round(price, dp),
    stochK: round(k, 1),
    stochD: round(d, 1),
    stochState,
    rsi: round(rsiV, 1),
    rsiAbove50: rsiV > 50,
    macdHist: round(hist, 4),
    macdCrossUp,
    macdCrossDown,
    entry: entry != null ? round(entry, dp) : null,
    stop: stop != null ? round(stop, dp) : null,
    target: target != null ? round(target, dp) : null,
    rr: 1.5,
    stopPct: stopPct != null ? round(stopPct * 100, 2) : null,
    reasons,
  };
}

/** Per-timeframe max-stop cap (forex 0.15% adapted up for equities). */
export function maxStopPctFor(tf: Timeframe): number {
  if (tf === "15m") return 0.012;
  if (tf === "1h") return 0.02;
  if (tf === "4h") return 0.03;
  return 0.04; // 1d / BTST
}
