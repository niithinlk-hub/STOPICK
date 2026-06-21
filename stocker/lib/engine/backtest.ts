/**
 * Walk-forward breakout backtest with Monte Carlo trade-sequence resampling.
 * Ported from STOPICK backtest/engine.py (next-bar execution, ATR stop, 2R
 * target, friction in bps). Uses a seeded PRNG so results are reproducible.
 */
import type { Bar } from "./types";
import { atr, closes, ema, mean, round, std, volumes } from "./indicators";

export interface BacktestTrade {
  entryTime: number;
  exitTime: number;
  entryPrice: number;
  exitPrice: number;
  stop: number;
  target: number;
  returnPct: number;
  rMultiple: number;
  exitReason: string;
  barsHeld: number;
}

export interface BacktestSummary {
  trades: number;
  winRate: number;
  expectancyR: number;
  profitFactor: number;
  maxDrawdownPct: number;
  cagrLikePct: number;
  sharpeLike: number;
  sortinoLike: number;
}

export interface WalkForwardRow {
  segment: string;
  trades: number;
  winRate: number;
  expectancyR: number;
}

export interface BacktestResult {
  trades: BacktestTrade[];
  equityCurve: { index: number; equity: number }[];
  summary: BacktestSummary;
  walkForward: WalkForwardRow[];
  monteCarlo: { run: number; endingR: number; worstR: number }[];
  mcStats: { p5: number; p50: number; p95: number };
}

// Mulberry32 deterministic PRNG (seed 42, matching STOPICK's fixed seed intent).
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function breakoutEntries(bars: Bar[], lookback: number, buffer: number, minVolRatio: number): boolean[] {
  const c = closes(bars);
  const ema20 = ema(c, 20);
  const ema50 = ema(c, 50);
  const vol = volumes(bars);
  const entries: boolean[] = new Array(bars.length).fill(false);
  for (let i = 0; i < bars.length; i++) {
    if (i < lookback || i < 20) continue;
    const level = Math.max(...bars.slice(i - lookback, i).map((b) => b.high));
    const avgVol = mean(vol.slice(i - 20, i));
    entries[i] =
      bars[i].close > level * (1 + buffer / 100) &&
      bars[i].volume > avgVol * minVolRatio &&
      bars[i].close > ema20[i] &&
      bars[i].close > ema50[i];
  }
  return entries;
}

export function runBacktest(
  bars: Bar[],
  opts: {
    lookback?: number;
    buffer?: number;
    minVolRatio?: number;
    slippageBps?: number;
    brokerageBps?: number;
    taxesBps?: number;
    maxHoldingBars?: number;
  } = {},
): BacktestResult {
  const lookback = opts.lookback ?? 20;
  const buffer = opts.buffer ?? 0.5;
  const minVolRatio = opts.minVolRatio ?? 1.5;
  const slippageBps = opts.slippageBps ?? 5;
  const brokerageBps = opts.brokerageBps ?? 3;
  const taxesBps = opts.taxesBps ?? 2;
  const maxHolding = opts.maxHoldingBars ?? 20;

  const atr14 = atr(bars, 14);
  const entries = breakoutEntries(bars, lookback, buffer, minVolRatio);
  const trades: BacktestTrade[] = [];

  let idx = 0;
  while (idx < bars.length - 2) {
    if (!entries[idx]) {
      idx++;
      continue;
    }
    const entryIdx = idx + 1;
    if (entryIdx >= bars.length) break;
    const entryPrice = bars[entryIdx].open * (1 + slippageBps / 10000);
    const atrValue = Number.isFinite(atr14[idx]) ? atr14[idx] : entryPrice * 0.03;
    const stop = entryPrice - atrValue * 1.5;
    const target = entryPrice + (entryPrice - stop) * 2;
    let exitIdx = Math.min(bars.length - 1, entryIdx + maxHolding);
    let exitPrice = bars[exitIdx].close;
    let exitReason = "time_stop";
    for (let probe = entryIdx; probe <= Math.min(bars.length - 1, entryIdx + maxHolding); probe++) {
      if (bars[probe].low <= stop) {
        exitPrice = stop * (1 - slippageBps / 10000);
        exitReason = "stop";
        exitIdx = probe;
        break;
      }
      if (bars[probe].high >= target) {
        exitPrice = target * (1 - slippageBps / 10000);
        exitReason = "target_2r";
        exitIdx = probe;
        break;
      }
    }
    const gross = exitPrice / entryPrice - 1;
    const totalCostBps = slippageBps * 2 + brokerageBps + taxesBps;
    const net = gross - totalCostBps / 10000;
    const rMultiple = (exitPrice - entryPrice) / Math.max(entryPrice - stop, 1e-9);
    trades.push({
      entryTime: bars[entryIdx].time,
      exitTime: bars[exitIdx].time,
      entryPrice: round(entryPrice, 4),
      exitPrice: round(exitPrice, 4),
      stop: round(stop, 4),
      target: round(target, 4),
      returnPct: round(net * 100, 4),
      rMultiple: round(rMultiple, 4),
      exitReason,
      barsHeld: exitIdx - entryIdx,
    });
    idx = exitIdx + 1;
  }

  const emptySummary: BacktestSummary = {
    trades: 0,
    winRate: 0,
    expectancyR: 0,
    profitFactor: 0,
    maxDrawdownPct: 0,
    cagrLikePct: 0,
    sharpeLike: 0,
    sortinoLike: 0,
  };

  if (!trades.length) {
    return {
      trades,
      equityCurve: [],
      summary: emptySummary,
      walkForward: [
        { segment: "in_sample", trades: 0, winRate: 0, expectancyR: 0 },
        { segment: "out_of_sample", trades: 0, winRate: 0, expectancyR: 0 },
      ],
      monteCarlo: [],
      mcStats: { p5: 0, p50: 0, p95: 0 },
    };
  }

  const returns = trades.map((t) => t.returnPct);
  let equity = 1;
  let peak = 1;
  let maxDd = 0;
  const equityCurve = trades.map((t, i) => {
    equity *= 1 + t.returnPct / 100;
    peak = Math.max(peak, equity);
    maxDd = Math.min(maxDd, equity / peak - 1);
    return { index: i, equity: round(equity, 4) };
  });

  const wins = returns.filter((r) => r > 0);
  const losses = returns.filter((r) => r <= 0);
  const grossWin = wins.reduce((a, b) => a + b, 0);
  const grossLoss = Math.abs(losses.reduce((a, b) => a + b, 0));
  const retStd = std(returns, 0);
  const negStd = std(losses, 0);
  const summary: BacktestSummary = {
    trades: trades.length,
    winRate: round((wins.length / trades.length) * 100, 2),
    expectancyR: round(mean(trades.map((t) => t.rMultiple)), 4),
    profitFactor: grossLoss > 0 ? round(grossWin / grossLoss, 4) : 0,
    maxDrawdownPct: round(maxDd * 100, 2),
    cagrLikePct: round((equity ** (252 / Math.max(bars.length, 1)) - 1) * 100, 2),
    sharpeLike: retStd ? round((mean(returns) / retStd) * Math.sqrt(12), 4) : 0,
    sortinoLike: negStd ? round((mean(returns) / negStd) * Math.sqrt(12), 4) : 0,
  };

  const split = Math.floor(bars.length * 0.7);
  const walkForward: WalkForwardRow[] = [];
  for (const [label, segment] of [
    ["in_sample", bars.slice(0, split)],
    ["out_of_sample", bars.slice(split)],
  ] as const) {
    if (segment.length <= 50) {
      walkForward.push({ segment: label, trades: 0, winRate: 0, expectancyR: 0 });
      continue;
    }
    const segEntries = breakoutEntries(segment, lookback, buffer, minVolRatio);
    const rate = mean(segEntries.map(Number));
    walkForward.push({
      segment: label,
      trades: segEntries.filter(Boolean).length,
      winRate: round(rate * 100, 2),
      expectancyR: round(rate * 2 - (1 - rate), 4),
    });
  }

  const rng = mulberry32(42);
  const tradeR = trades.map((t) => t.rMultiple);
  const monteCarlo = Array.from({ length: 200 }, (_, run) => {
    let sum = 0;
    let worst = Infinity;
    for (let k = 0; k < tradeR.length; k++) {
      const v = tradeR[Math.floor(rng() * tradeR.length)];
      sum += v;
      worst = Math.min(worst, v);
    }
    return { run, endingR: round(sum, 4), worstR: round(worst === Infinity ? 0 : worst, 4) };
  });
  const endings = monteCarlo.map((m) => m.endingR).sort((a, b) => a - b);
  const pct = (p: number) => endings[Math.min(endings.length - 1, Math.floor((p / 100) * endings.length))];
  const mcStats = { p5: round(pct(5), 2), p50: round(pct(50), 2), p95: round(pct(95), 2) };

  return { trades, equityCurve, summary, walkForward, monteCarlo, mcStats };
}
