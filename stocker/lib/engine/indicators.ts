/**
 * Technical indicators. Ported from STOPICK's signals/common.py with pandas
 * semantics preserved (ewm adjust=False, Wilder smoothing via alpha=1/period,
 * rolling means with NaN warm-up). Series are plain number[] aligned to bars;
 * NaN marks "not enough data yet".
 */
import type { Bar } from "./types";

export const closes = (bars: Bar[]): number[] => bars.map((b) => b.close);
export const highs = (bars: Bar[]): number[] => bars.map((b) => b.high);
export const lows = (bars: Bar[]): number[] => bars.map((b) => b.low);
export const volumes = (bars: Bar[]): number[] => bars.map((b) => b.volume);

/** Last finite value of a series, or a fallback. */
export function last(series: number[], fallback = NaN): number {
  for (let i = series.length - 1; i >= 0; i--) {
    if (Number.isFinite(series[i])) return series[i];
  }
  return fallback;
}

export function mean(values: number[]): number {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return NaN;
  return finite.reduce((a, b) => a + b, 0) / finite.length;
}

export function std(values: number[], ddof = 0): number {
  const finite = values.filter((v) => Number.isFinite(v));
  const n = finite.length;
  if (n - ddof <= 0) return NaN;
  const m = finite.reduce((a, b) => a + b, 0) / n;
  const variance = finite.reduce((a, b) => a + (b - m) ** 2, 0) / (n - ddof);
  return Math.sqrt(variance);
}

/** EMA (pandas ewm adjust=False), seeded at the first value. No NaN warm-up. */
export function ema(values: number[], span: number): number[] {
  const alpha = 2 / (span + 1);
  const out: number[] = new Array(values.length).fill(NaN);
  let prev = NaN;
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    if (!Number.isFinite(x)) {
      out[i] = prev;
      continue;
    }
    prev = Number.isFinite(prev) ? alpha * x + (1 - alpha) * prev : x;
    out[i] = prev;
  }
  return out;
}

export function sma(values: number[], period: number): number[] {
  const out: number[] = new Array(values.length).fill(NaN);
  let sum = 0;
  const window: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    window.push(x);
    sum += Number.isFinite(x) ? x : 0;
    if (window.length > period) sum -= Number.isFinite(window.shift()!) ? (window as number[])[0] : 0;
    if (window.length >= period) out[i] = sum / period;
  }
  return out;
}

/** Wilder smoothing (ewm alpha=1/period) with a `period`-bar warm-up of NaN. */
function wilder(values: number[], period: number): number[] {
  const alpha = 1 / period;
  const out: number[] = new Array(values.length).fill(NaN);
  let prev = NaN;
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    const x = Number.isFinite(values[i]) ? values[i] : 0;
    prev = Number.isFinite(prev) ? alpha * x + (1 - alpha) * prev : x;
    count++;
    if (count >= period) out[i] = prev;
  }
  return out;
}

function trueRange(bars: Bar[]): number[] {
  return bars.map((b, i) => {
    if (i === 0) return b.high - b.low;
    const pc = bars[i - 1].close;
    return Math.max(b.high - b.low, Math.abs(b.high - pc), Math.abs(b.low - pc));
  });
}

/** ATR = simple rolling mean of true range (pandas rolling mean, min_periods=period). */
export function atr(bars: Bar[], period = 14): number[] {
  return sma(trueRange(bars), period);
}

/** RSI via Wilder smoothing. NaN until `period` deltas exist. */
export function rsi(values: number[], period = 14): number[] {
  const gains: number[] = [0];
  const losses: number[] = [0];
  for (let i = 1; i < values.length; i++) {
    const delta = values[i] - values[i - 1];
    gains.push(delta > 0 ? delta : 0);
    losses.push(delta < 0 ? -delta : 0);
  }
  const avgGain = wilder(gains, period);
  const avgLoss = wilder(losses, period);
  return values.map((_, i) => {
    const g = avgGain[i];
    const l = avgLoss[i];
    if (!Number.isFinite(g) || !Number.isFinite(l)) return NaN;
    if (l === 0) return 100;
    const rs = g / l;
    return 100 - 100 / (1 + rs);
  });
}

export interface MacdSeries {
  macd: number[];
  signal: number[];
  histogram: number[];
}

/** MACD(12,26,9). STOCKER addition (not present in STOPICK). */
export function macd(values: number[], fast = 12, slow = 26, signalPeriod = 9): MacdSeries {
  const emaFast = ema(values, fast);
  const emaSlow = ema(values, slow);
  const macdLine = values.map((_, i) => emaFast[i] - emaSlow[i]);
  const signal = ema(macdLine, signalPeriod);
  const histogram = macdLine.map((v, i) => v - signal[i]);
  return { macd: macdLine, signal, histogram };
}

export interface StochSeries {
  k: number[];
  d: number[];
}

/**
 * Classic Stochastic oscillator (the one used in the RSI-confluence strategy —
 * NOT Stochastic-RSI). raw %K = (close - lowestLow)/(highestHigh - lowestLow)*100
 * over `kPeriod`; %K is then smoothed by `kSmooth` (slow stochastic) and %D is an
 * SMA of %K over `dSmooth`. Defaults 14/3/3.
 */
export function stochastic(bars: Bar[], kPeriod = 14, kSmooth = 3, dSmooth = 3): StochSeries {
  const raw: number[] = new Array(bars.length).fill(NaN);
  for (let i = 0; i < bars.length; i++) {
    if (i < kPeriod - 1) continue;
    const window = bars.slice(i - kPeriod + 1, i + 1);
    const hi = Math.max(...window.map((b) => b.high));
    const lo = Math.min(...window.map((b) => b.low));
    const denom = hi - lo;
    raw[i] = denom > 0 ? ((bars[i].close - lo) / denom) * 100 : 50;
  }
  const k = sma(raw, kSmooth);
  const d = sma(k, dSmooth);
  return { k, d };
}

/** ADX (directional movement index) — ported from common.adx. */
export function adx(bars: Bar[], period = 14): number[] {
  const n = bars.length;
  const plusDM: number[] = new Array(n).fill(0);
  const minusDM: number[] = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const up = bars[i].high - bars[i - 1].high;
    const down = bars[i - 1].low - bars[i].low;
    plusDM[i] = up > down && up > 0 ? up : 0;
    minusDM[i] = down > up && down > 0 ? down : 0;
  }
  const tr = wilder(trueRange(bars), period);
  const plusSm = wilder(plusDM, period);
  const minusSm = wilder(minusDM, period);
  const dx: number[] = new Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const t = tr[i];
    if (!Number.isFinite(t) || t === 0) continue;
    const plusDI = (100 * plusSm[i]) / t;
    const minusDI = (100 * minusSm[i]) / t;
    const denom = plusDI + minusDI;
    if (denom === 0) {
      dx[i] = 0;
      continue;
    }
    dx[i] = (Math.abs(plusDI - minusDI) / denom) * 100;
  }
  return wilder(dx, period);
}

/** On-balance volume — cumulative sign(close diff) * volume. */
export function obv(bars: Bar[]): number[] {
  const out: number[] = new Array(bars.length).fill(0);
  let cum = 0;
  for (let i = 1; i < bars.length; i++) {
    const dir = Math.sign(bars[i].close - bars[i - 1].close);
    cum += dir * bars[i].volume;
    out[i] = cum;
  }
  return out;
}

/** Rolling percentile rank (0..1) of the last value within a trailing window. */
export function rollingPercentile(series: number[], window = 252): number[] {
  const minPeriods = Math.max(20, Math.floor(window / 5));
  const out: number[] = new Array(series.length).fill(NaN);
  for (let i = 0; i < series.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = series.slice(start, i + 1).filter((v) => Number.isFinite(v));
    if (slice.length < minPeriods) continue;
    const current = series[i];
    if (!Number.isFinite(current)) continue;
    const below = slice.filter((v) => v <= current).length;
    out[i] = below / slice.length;
  }
  return out;
}

/** Linear-regression slope of the last `window` finite values. */
export function linregSlope(series: number[], window = 20): number {
  const cleaned = series.filter((v) => Number.isFinite(v)).slice(-window);
  if (cleaned.length < 5) return 0;
  const n = cleaned.length;
  const xs = cleaned.map((_, i) => i);
  const sx = xs.reduce((a, b) => a + b, 0);
  const sy = cleaned.reduce((a, b) => a + b, 0);
  const sxx = xs.reduce((a, b) => a + b * b, 0);
  const sxy = xs.reduce((a, x, i) => a + x * cleaned[i], 0);
  const denom = n * sxx - sx * sx;
  if (denom === 0) return 0;
  return (n * sxy - sx * sy) / denom;
}

/** Kaufman efficiency ratio over the last `window` finite values. */
export function efficiencyRatio(series: number[], window = 20): number {
  const cleaned = series.filter((v) => Number.isFinite(v)).slice(-window);
  if (cleaned.length < 5) return 0;
  const netChange = Math.abs(cleaned[cleaned.length - 1] - cleaned[0]);
  let path = 0;
  for (let i = 1; i < cleaned.length; i++) path += Math.abs(cleaned[i] - cleaned[i - 1]);
  return path ? netChange / path : 0;
}

export const bounded = (v: number, lo = 0, hi = 1): number => Math.max(lo, Math.min(hi, v));
export const round = (v: number, dp = 2): number => {
  if (!Number.isFinite(v)) return 0;
  const f = 10 ** dp;
  return Math.round(v * f) / f;
};
