/**
 * Context engine: volume participation, relative strength (with sector routing),
 * and market regime. Ported from STOPICK signals/context.py and extended with
 * an above-anchored-VWAP flag, an OBV accumulation trend, and sector alpha.
 */
import type {
  Bar,
  Direction,
  Market,
  RegimeSignal,
  RelativeStrengthSignal,
  VolumeSignal,
} from "./types";
import {
  atr,
  bounded,
  closes,
  ema,
  efficiencyRatio,
  last,
  linregSlope,
  mean,
  obv,
  rollingPercentile,
  round,
} from "./indicators";

export function analyzeVolumeParticipation(bars: Bar[], breakoutLevel: number | null = null): VolumeSignal {
  const latest = bars[bars.length - 1];
  const tail21 = bars.slice(-21);
  const avg20 = tail21.length >= 21 ? mean(tail21.slice(0, -1).map((b) => b.volume)) : mean(bars.map((b) => b.volume));
  const volumeRatio = avg20 ? latest.volume / avg20 : 1;

  const obvSeries = obv(bars);
  const obvConfirmation = obvSeries.length > 5 && obvSeries[obvSeries.length - 1] >= obvSeries[obvSeries.length - 6];
  const obvSlope = linregSlope(obvSeries.slice(-20), 20);
  const accumulationTrend = Math.sign(obvSlope);

  // Rolling 20-bar VWAP.
  const typical = bars.map((b) => (b.high + b.low + b.close) / 3);
  let vwapAlignment = false;
  {
    const w = 20;
    const slice = bars.slice(-w);
    if (slice.length >= 5) {
      const num = slice.reduce((a, b) => a + ((b.high + b.low + b.close) / 3) * b.volume, 0);
      const den = slice.reduce((a, b) => a + b.volume, 0);
      const vwap = den ? num / den : NaN;
      vwapAlignment = Number.isFinite(vwap) && latest.close >= vwap;
    }
  }

  // Anchored VWAP from the first close above the breakout level (else last 20 bars).
  let anchorIdx = Math.max(bars.length - 20, 0);
  if (breakoutLevel !== null) {
    const found = bars.findIndex((b) => b.close >= breakoutLevel);
    if (found >= 0) anchorIdx = found;
  }
  const anchored = bars.slice(anchorIdx);
  const anchoredDen = anchored.reduce((a, b) => a + b.volume, 0);
  const anchoredVwap = anchoredDen
    ? anchored.reduce((a, b) => a + ((b.high + b.low + b.close) / 3) * b.volume, 0) / anchoredDen
    : null;
  const aboveAnchoredVwap = anchoredVwap !== null && latest.close >= anchoredVwap;

  const last5 = mean(bars.slice(-5).map((b) => b.volume));
  const last20 = mean(bars.slice(-20).map((b) => b.volume));
  const dryUp = bars.length >= 11 && last5 < last20 * 0.85;

  const penaltyFlags: string[] = [];
  if (volumeRatio < 1.2) penaltyFlags.push("breakout_without_participation");
  const spread = latest.high - latest.low;
  const atrValue = last(atr(bars, 14), 0) || 0;
  if (atrValue && spread > atrValue * 1.5 && volumeRatio < 1.3) penaltyFlags.push("wide_spread_weak_volume");
  if (volumeRatio > 3 && latest.close < latest.high * 0.98) penaltyFlags.push("possible_exhaustion");

  return {
    volumeRatio: round(volumeRatio, 2),
    relativeVolume: round(volumeRatio, 2),
    obvConfirmation,
    vwapAlignment,
    anchoredVwap: anchoredVwap !== null ? round(anchoredVwap, 4) : null,
    aboveAnchoredVwap,
    dryUpBeforeExpansion: dryUp,
    accumulationTrend,
    penaltyFlags,
  };
}

function alignClose(asset: Bar[], benchmark: Bar[]): { a: number[]; b: number[] } {
  const benchMap = new Map<number, number>();
  for (const bar of benchmark) benchMap.set(bar.time, bar.close);
  const a: number[] = [];
  const b: number[] = [];
  for (const bar of asset) {
    const bc = benchMap.get(bar.time);
    if (bc !== undefined) {
      a.push(bar.close);
      b.push(bc);
    }
  }
  return { a, b };
}

export function analyzeRelativeStrength(
  bars: Bar[],
  benchmarkBars: Bar[],
  opts: { benchmarkSymbol: string; sectorBars?: Bar[] | null; sectorSymbol?: string | null },
): RelativeStrengthSignal {
  const { a, b } = alignClose(bars, benchmarkBars);
  if (!a.length) {
    return {
      benchmarkSymbol: opts.benchmarkSymbol,
      sectorBenchmarkSymbol: opts.sectorSymbol ?? null,
      score: 0,
      trendPersistence: 0,
      smoothness: 0,
      oneWeekAlpha: 0,
      oneMonthAlpha: 0,
      threeMonthAlpha: 0,
      sectorAlpha: null,
      explanation: "Benchmark overlap was unavailable.",
    };
  }

  const alpha = (window: number): number => {
    if (a.length < window + 1) return 0;
    const ar = a[a.length - 1] / a[a.length - 1 - window] - 1;
    const br = b[b.length - 1] / b[b.length - 1 - window] - 1;
    return ar - br;
  };

  const oneWeek = alpha(5);
  const oneMonth = alpha(21);
  const threeMonth = alpha(63);
  const rsLine = a.map((v, i) => v / b[i]);
  const smoothness = efficiencyRatio(rsLine, 20);
  let ups = 0;
  let total = 0;
  for (let i = Math.max(1, rsLine.length - 20); i < rsLine.length; i++) {
    if (rsLine[i] - rsLine[i - 1] > 0) ups++;
    total++;
  }
  const persistence = total ? ups / total : 0;

  let sectorAlpha: number | null = null;
  if (opts.sectorBars && opts.sectorBars.length) {
    const { a: sa, b: sb } = alignClose(bars, opts.sectorBars);
    if (sa.length > 22) {
      const ar = sa[sa.length - 1] / sa[sa.length - 22] - 1;
      const sr = sb[sb.length - 1] / sb[sb.length - 22] - 1;
      sectorAlpha = round((ar - sr) * 100, 2);
    }
  }

  const raw = (oneWeek * 20 + oneMonth * 35 + threeMonth * 45 + smoothness * 20 + persistence * 10) * 10;
  const score = Math.max(0, Math.min(100, raw));

  return {
    benchmarkSymbol: opts.benchmarkSymbol,
    sectorBenchmarkSymbol: opts.sectorSymbol ?? null,
    score: round(score, 2),
    trendPersistence: round(persistence * 100, 2),
    smoothness: round(smoothness * 100, 2),
    oneWeekAlpha: round(oneWeek * 100, 2),
    oneMonthAlpha: round(oneMonth * 100, 2),
    threeMonthAlpha: round(threeMonth * 100, 2),
    sectorAlpha,
    explanation:
      "Relative strength is measured against the selected benchmark with 1W/1M/3M alpha and path smoothness.",
  };
}

export function analyzeMarketRegime(benchmarkBars: Bar[], opts: { market: Market; benchmarkSymbol: string }): RegimeSignal {
  if (benchmarkBars.length < 220) {
    return {
      market: opts.market,
      benchmarkSymbol: opts.benchmarkSymbol,
      direction: "neutral",
      trendStrength: 0,
      volatilityState: "unknown",
      breadthLikeProxy: 0,
      explanation: "Insufficient benchmark history.",
    };
  }
  const c = closes(benchmarkBars);
  const ema20 = ema(c, 20);
  const ema50 = ema(c, 50);
  const ema200 = ema(c, 200);
  const atrSeries = atr(benchmarkBars, 14);
  const atrPctile = rollingPercentile(atrSeries, 252);
  const i = benchmarkBars.length - 1;
  const close = c[i];

  const bullish = close > ema20[i] && ema20[i] > ema50[i] && ema50[i] > ema200[i];
  const bearish = close < ema20[i] && ema20[i] < ema50[i] && ema50[i] < ema200[i];
  const direction: Direction = bullish ? "bullish" : bearish ? "bearish" : "neutral";

  const trendStrength =
    ([close > ema20[i], ema20[i] > ema50[i], ema50[i] > ema200[i]].map(Number).reduce((a, b) => a + b, 0) / 3) * 100;
  const atrP = Number.isFinite(atrPctile[i]) ? atrPctile[i] : 0.5;
  const volatilityState = atrP > 0.65 ? "expanding" : atrP < 0.35 ? "compressed" : "normal";

  let above = 0;
  const last50 = benchmarkBars.slice(-50);
  const startIdx = benchmarkBars.length - last50.length;
  for (let k = 0; k < last50.length; k++) above += last50[k].close > ema20[startIdx + k] ? 1 : 0;
  const breadthProxy = last50.length ? (above / last50.length) * 100 : 0;

  return {
    market: opts.market,
    benchmarkSymbol: opts.benchmarkSymbol,
    direction,
    trendStrength: round(trendStrength, 2),
    volatilityState,
    breadthLikeProxy: round(breadthProxy, 2),
    explanation: `${opts.market} regime is ${direction} with ${volatilityState} volatility relative to ${opts.benchmarkSymbol}.`,
  };
}
