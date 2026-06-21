/**
 * Multi-timeframe trend alignment. Ported from STOPICK signals/trend_alignment.py
 * and extended with an explicit mtfAgreement score.
 */
import type { Bar, Direction, TrendSignal } from "./types";
import { adx, closes, ema, highs, last, linregSlope, lows } from "./indicators";

function hhHlScore(bars: Bar[]): number {
  const recent = bars.slice(-40);
  if (recent.length < 20) return 0;
  const rollMax: number[] = [];
  const rollMin: number[] = [];
  for (let i = 4; i < recent.length; i++) {
    rollMax.push(Math.max(...recent.slice(i - 4, i + 1).map((b) => b.high)));
    rollMin.push(Math.min(...recent.slice(i - 4, i + 1).map((b) => b.low)));
  }
  if (rollMax.length < 4 || rollMin.length < 4) return 0;
  const hh = rollMax[rollMax.length - 1] > rollMax[rollMax.length - 4];
  const hl = rollMin[rollMin.length - 1] > rollMin[rollMin.length - 4];
  const lh = rollMax[rollMax.length - 1] < rollMax[rollMax.length - 4];
  const ll = rollMin[rollMin.length - 1] < rollMin[rollMin.length - 4];
  if (hh && hl) return 1;
  if (lh && ll) return -1;
  return 0;
}

interface TfResult {
  direction: Direction;
  score: number;
  metrics: Record<string, number | string | null>;
}

export function analyzeTimeframeTrend(bars: Bar[], timeframe: string, adxThreshold = 20): TfResult {
  if (bars.length < 220) return { direction: "neutral", score: 0, metrics: { timeframe } };
  const c = closes(bars);
  const ema20 = ema(c, 20);
  const ema50 = ema(c, 50);
  const ema200 = ema(c, 200);
  const adx14 = adx(bars, 14);
  const i = bars.length - 1;
  const close = c[i];
  const hhhl = hhHlScore(bars);
  const ema20Slope = linregSlope(ema20, 15);
  const adxVal = Number.isFinite(adx14[i]) ? adx14[i] : 0;

  const bullChecks = [
    ema20[i] > ema50[i] && ema50[i] > ema200[i],
    close > ema20[i] && close > ema50[i],
    hhhl > 0,
    adxVal >= adxThreshold,
    ema20Slope > 0,
  ].map(Number);
  const bearChecks = [
    ema20[i] < ema50[i] && ema50[i] < ema200[i],
    close < ema20[i] && close < ema50[i],
    hhhl < 0,
    adxVal >= adxThreshold,
    ema20Slope < 0,
  ].map(Number);

  const bull = bullChecks.reduce((a, b) => a + b, 0) / bullChecks.length;
  const bear = bearChecks.reduce((a, b) => a + b, 0) / bearChecks.length;

  let direction: Direction;
  let score: number;
  if (bull > bear && bull >= 0.6) {
    direction = "bullish";
    score = bull;
  } else if (bear > bull && bear >= 0.6) {
    direction = "bearish";
    score = bear;
  } else {
    direction = "neutral";
    score = Math.max(bull, bear) * 0.5;
  }

  return {
    direction,
    score,
    metrics: {
      timeframe,
      close,
      ema20: ema20[i],
      ema50: ema50[i],
      ema200: ema200[i],
      adx14: Number.isFinite(adx14[i]) ? adx14[i] : null,
      ema20_slope: ema20Slope,
      hh_hl_state: hhhl,
    },
  };
}

export function analyzeTrendAlignment(
  frameMap: Record<string, Bar[]>,
  opts: { adxThreshold?: number; timeframes?: string[] } = {},
): TrendSignal {
  const adxThreshold = opts.adxThreshold ?? 20;
  const timeframes = opts.timeframes ?? ["1d", "4h", "1h", "15m"];
  const scores: Record<string, number> = {};
  const metrics: Record<string, number | string | null> = {};
  const directions: Record<string, Direction> = {};
  let bullVotes = 0;
  let bearVotes = 0;

  for (const tf of timeframes) {
    const bars = frameMap[tf];
    if (!bars || bars.length === 0) continue;
    const { direction, score, metrics: m } = analyzeTimeframeTrend(bars, tf, adxThreshold);
    scores[tf] = score;
    directions[tf] = direction;
    for (const [k, v] of Object.entries(m)) metrics[`${tf}_${k}`] = v;
    if (direction === "bullish") bullVotes++;
    else if (direction === "bearish") bearVotes++;
  }

  const scored = Object.keys(scores).length;
  let direction: Direction = "neutral";
  if (bullVotes > bearVotes && scored) direction = "bullish";
  else if (bearVotes > bullVotes && scored) direction = "bearish";

  const alignmentConfidence = (Math.max(bullVotes, bearVotes) / Math.max(scored, 1)) * 100;
  const scoreVals = Object.values(scores);
  const strengthScore = (scoreVals.reduce((a, b) => a + b, 0) / Math.max(scored, 1)) * 100;

  // mtfAgreement: of the timeframes that voted for the dominant direction, weight
  // by their mean strength — rewards stacks that are both aligned and strong.
  const agreeing = Object.entries(directions).filter(([, d]) => d === direction && d !== "neutral");
  const agreeMeanStrength = agreeing.length
    ? agreeing.reduce((a, [tf]) => a + scores[tf], 0) / agreeing.length
    : 0;
  const mtfAgreement = scored ? (agreeing.length / scored) * agreeMeanStrength * 100 : 0;

  const tfScores: Record<string, number> = {};
  for (const [tf, s] of Object.entries(scores)) tfScores[tf] = Math.round(s * 100 * 100) / 100;

  return {
    direction,
    strengthScore: Math.round(strengthScore * 100) / 100,
    alignmentConfidence: Math.round(alignmentConfidence * 100) / 100,
    timeframeScores: tfScores,
    mtfAgreement: Math.round(mtfAgreement * 100) / 100,
    metrics,
  };
}
