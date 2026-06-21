/**
 * Breakout pattern detection. Ported from STOPICK signals/breakout.py (base, ATH,
 * flag, cup-and-handle, VCP, event continuation) and extended with a
 * follow-through (false-breakout) guard and ATR extension measure.
 */
import type { Bar, BreakoutSignal, Direction, Market, StructureSignal, TrendSignal } from "./types";
import { atr, bounded, closes, ema, last, mean, round, std } from "./indicators";

interface Candidate {
  patternName: string;
  isValid: boolean;
  breakoutLevel: number | null;
  bufferedLevel: number | null;
  invalidationLevel: number | null;
  touches: number;
}

function candleCloseQuality(b: Bar): number {
  const totalRange = b.high - b.low;
  if (totalRange <= 0) return 0;
  const body = Math.abs(b.close - b.open);
  const closeNearHigh = 1 - (b.high - b.close) / totalRange;
  return bounded((body / totalRange) * 0.55 + closeNearHigh * 0.45);
}

function tightnessScore(bars: Bar[], lookback = 20): number {
  const recent = bars.slice(-lookback);
  if (!recent.length) return 0;
  const c = recent.map((b) => b.close);
  const compression = std(c) / Math.max(mean(c), 1e-9);
  return round(bounded(1 - compression * 8), 4);
}

function volumeRatio(bars: Bar[]): number {
  const recent = bars.slice(-21);
  if (recent.length < 21) return 1;
  const avg = mean(recent.slice(0, -1).map((b) => b.volume));
  return avg ? recent[recent.length - 1].volume / avg : 1;
}

function distancePct(current: number, level: number | null): number | null {
  if (level === null || level === 0) return null;
  return round((current / level - 1) * 100, 4);
}

function overheadResistancePct(bars: Bar[], current: number): number | null {
  const window = bars.slice(-252).map((b) => b.high);
  if (!window.length) return null;
  const overhead = Math.max(...window);
  if (overhead <= current) return null;
  return round((overhead / current - 1) * 100, 4);
}

function baseBreakout(bars: Bar[], buffer: number, lookback: number): Candidate | null {
  if (bars.length < lookback + 5) return null;
  const baseSlice = bars.slice(0, -1).slice(-lookback);
  const level = Math.max(...baseSlice.map((b) => b.high));
  const lastClose = bars[bars.length - 1].close;
  const buffered = level * (1 + buffer / 100);
  return {
    patternName: "Base breakout",
    isValid: lastClose > buffered,
    breakoutLevel: level,
    bufferedLevel: buffered,
    invalidationLevel: Math.min(...baseSlice.map((b) => b.low)),
    touches: baseSlice.filter((b) => b.high >= level * 0.995).length,
  };
}

function athBreakout(bars: Bar[], buffer: number): Candidate | null {
  if (bars.length < 260) return null;
  const prevHigh = Math.max(...bars.slice(0, -1).slice(-252).map((b) => b.high));
  const lastClose = bars[bars.length - 1].close;
  const buffered = prevHigh * (1 + buffer / 100);
  return {
    patternName: "ATH breakout",
    isValid: lastClose > buffered,
    breakoutLevel: prevHigh,
    bufferedLevel: buffered,
    invalidationLevel: Math.min(...bars.slice(-20).map((b) => b.low)),
    touches: 1,
  };
}

function flagBreakout(bars: Bar[], buffer: number): Candidate | null {
  if (bars.length < 35) return null;
  const impulse = bars.slice(-35, -15);
  const flag = bars.slice(-15, -1);
  if (!impulse.length || !flag.length) return null;
  const impulseGain = impulse[impulse.length - 1].close / impulse[0].close - 1;
  const flagDepth = (Math.max(...flag.map((b) => b.high)) - Math.min(...flag.map((b) => b.low))) /
    Math.max(impulse[impulse.length - 1].close, 1e-9);
  const level = Math.max(...flag.map((b) => b.high));
  const lastClose = bars[bars.length - 1].close;
  const buffered = level * (1 + buffer / 100);
  return {
    patternName: "Flag continuation",
    isValid: impulseGain > 0.08 && flagDepth < 0.08 && lastClose > buffered,
    breakoutLevel: level,
    bufferedLevel: buffered,
    invalidationLevel: Math.min(...flag.map((b) => b.low)),
    touches: 2,
  };
}

function cupHandle(bars: Bar[], buffer: number): Candidate | null {
  if (bars.length < 80) return null;
  const cup = bars.slice(-80, -15);
  const handle = bars.slice(-15, -1);
  if (!cup.length || !handle.length) return null;
  const rim = Math.max(cup[0].high, cup[cup.length - 1].high);
  const trough = Math.min(...cup.map((b) => b.low));
  const recovery = (cup[cup.length - 1].close - trough) / Math.max(rim - trough, 1e-9);
  const handleDepth = (Math.max(...handle.map((b) => b.high)) - Math.min(...handle.map((b) => b.low))) / Math.max(rim, 1e-9);
  const buffered = rim * (1 + buffer / 100);
  return {
    patternName: "Cup-and-handle approximation",
    isValid: recovery > 0.8 && handleDepth < 0.08 && bars[bars.length - 1].close > buffered,
    breakoutLevel: rim,
    bufferedLevel: buffered,
    invalidationLevel: Math.min(...handle.map((b) => b.low)),
    touches: 2,
  };
}

function vcpBreakout(bars: Bar[], buffer: number, lookback: number): Candidate | null {
  if (bars.length < lookback + 10) return null;
  const recent = bars.slice(-lookback - 1, -1);
  const ranges: number[] = [];
  for (let i = 0; i < recent.length; i++) {
    const window = recent.slice(Math.max(0, i - 9), i + 1);
    if (window.length >= 3) ranges.push(mean(window.map((b) => b.high - b.low)));
  }
  const level = Math.max(...recent.map((b) => b.high));
  const contraction = ranges.length >= 2 && ranges[ranges.length - 1] < ranges[0] * 0.7;
  const buffered = level * (1 + buffer / 100);
  return {
    patternName: "Volatility contraction breakout",
    isValid: contraction && bars[bars.length - 1].close > buffered,
    breakoutLevel: level,
    bufferedLevel: buffered,
    invalidationLevel: Math.min(...recent.slice(-10).map((b) => b.low)),
    touches: recent.filter((b) => b.high >= level * 0.995).length,
  };
}

function eventContinuation(bars: Bar[], eventDays: number | null, market: Market, buffer: number): Candidate | null {
  if (eventDays === null) return null;
  const recent = bars.slice(-10);
  if (!recent.length) return null;
  const level = Math.max(...recent.slice(0, -1).map((b) => b.high));
  const buffered = level * (1 + buffer / 100);
  return {
    patternName: market === "US" ? "Earnings breakout continuation" : "Results breakout continuation",
    isValid: eventDays <= 5 && bars[bars.length - 1].close > buffered,
    breakoutLevel: level,
    bufferedLevel: buffered,
    invalidationLevel: Math.min(...recent.map((b) => b.low)),
    touches: 1,
  };
}

/**
 * Follow-through / false-breakout guard (STOCKER addition). Rewards a breakout
 * bar that closes near its high above the buffered level on expanding volume,
 * penalizes upper-wick rejections. 0..100.
 */
function followThrough(lastBar: Bar, buffered: number | null, volRatio: number, candleQ: number): number {
  const range = lastBar.high - lastBar.low;
  const upperWick = range > 0 ? (lastBar.high - Math.max(lastBar.open, lastBar.close)) / range : 0;
  const closeLocation = range > 0 ? (lastBar.close - lastBar.low) / range : 0.5;
  const clearsBuffer = buffered !== null ? bounded((lastBar.close / buffered - 1) / 0.01) : 0.5;
  const score =
    closeLocation * 0.35 +
    bounded((volRatio - 1) / 1.5) * 0.25 +
    candleQ * 0.2 +
    clearsBuffer * 0.2 -
    upperWick * 0.3;
  return round(bounded(score) * 100, 2);
}

export function findBestBreakout(
  bars: Bar[],
  opts: {
    market: Market;
    trend: TrendSignal;
    structure: StructureSignal;
    relativeStrengthScore: number;
    buffer?: number;
    lookback?: number;
    eventDays?: number | null;
  },
): BreakoutSignal {
  const buffer = opts.buffer ?? 0.5;
  const lookback = opts.lookback ?? 40;
  const eventDays = opts.eventDays ?? null;
  const lastBar = bars[bars.length - 1];
  const c = closes(bars);
  const ema20 = last(ema(c, 20));
  const atrVal = last(atr(bars, 14), 0);

  const candidates = [
    baseBreakout(bars, buffer, lookback),
    athBreakout(bars, buffer),
    flagBreakout(bars, buffer),
    cupHandle(bars, buffer),
    vcpBreakout(bars, buffer, Math.min(30, lookback)),
    eventContinuation(bars, eventDays, opts.market, buffer),
  ].filter((x): x is Candidate => x !== null);

  if (!candidates.length) {
    return {
      isValid: false,
      patternName: "None",
      direction: opts.trend.direction,
      breakoutLevel: null,
      bufferedLevel: null,
      currentPrice: lastBar.close,
      distancePct: null,
      candleQuality: 0,
      tightnessScore: 0,
      volumeExpansion: 1,
      overheadResistancePct: null,
      invalidationLevel: null,
      followThroughScore: 0,
      extensionAtr: null,
      explanation: "No breakout candidate pattern was available.",
      metrics: {},
    };
  }

  const volRatio = volumeRatio(bars);
  const candleQ = candleCloseQuality(lastBar);
  const tightness = tightnessScore(bars, 20);
  const overhead = overheadResistancePct(bars, lastBar.close);

  const best = candidates.reduce((a, b) => {
    const ka = [Number(a.isValid), a.touches];
    const kb = [Number(b.isValid), b.touches];
    if (kb[0] !== ka[0]) return kb[0] > ka[0] ? b : a;
    if (kb[1] !== ka[1]) return kb[1] > ka[1] ? b : a;
    return a;
  });

  let isValid = best.isValid && opts.trend.direction !== "bearish";
  const reasons = [
    `${best.patternName} uses breakout level ${(best.breakoutLevel ?? 0).toFixed(2)}.`,
    `Volume ratio is ${volRatio.toFixed(2)}x and candle quality is ${candleQ.toFixed(2)}.`,
    `Relative strength score is ${opts.relativeStrengthScore.toFixed(1)}.`,
    opts.structure.explanation,
  ];
  if (overhead !== null && overhead < 3) {
    reasons.push("There is nearby higher-timeframe resistance overhead.");
    isValid = false;
  }
  if (volRatio < 1.2 || candleQ < 0.55) {
    reasons.push("Breakout rejected: weak volume participation or poor close quality.");
    isValid = false; // gate, not just a downgrade — a breakout without participation fails often
  }

  let invalidation = best.invalidationLevel;
  if (invalidation === null && Number.isFinite(atrVal)) invalidation = lastBar.close - atrVal * 1.5;

  const ft = followThrough(lastBar, best.bufferedLevel, volRatio, candleQ);
  const extensionAtr = atrVal ? round((lastBar.close - ema20) / atrVal, 2) : null;

  return {
    isValid,
    patternName: best.patternName,
    direction: opts.trend.direction,
    breakoutLevel: best.breakoutLevel,
    bufferedLevel: best.bufferedLevel,
    currentPrice: lastBar.close,
    distancePct: distancePct(lastBar.close, best.breakoutLevel),
    candleQuality: round(candleQ * 100, 2),
    tightnessScore: round(tightness * 100, 2),
    volumeExpansion: round(volRatio, 2),
    overheadResistancePct: overhead,
    invalidationLevel: invalidation !== null && Number.isFinite(invalidation) ? invalidation : null,
    followThroughScore: ft,
    extensionAtr,
    explanation: reasons.join(" "),
    metrics: {
      touches: best.touches,
      relative_strength_score: opts.relativeStrengthScore,
      trend_direction: opts.trend.direction,
      bos_present: opts.structure.bos,
    },
  };
}
