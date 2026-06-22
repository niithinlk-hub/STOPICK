/**
 * Screening-engine validation harness (Algo Check). Walks daily history bar-by-bar with
 * NO lookahead — at each decision bar it runs the SAME signal functions + scoreSetup the
 * live scanner uses (on bars[0..t] only), enters at the next bar's open, then simulates
 * forward to a 2R target / ATR stop / time-exit. Results are bucketed by grade and score
 * so we can see whether a higher grade actually earns a higher expectancy (calibration).
 *
 * Isolated from runScan on purpose — the live Scanner tab is untouched. An optional
 * `profileOverride` lets Algo Check A/B-test experimental weights before any promotion.
 */
import type { Bar, Market, ScoringProfile, SetupSignal } from "./types";
import { CONFIG } from "../config";
import { analyzeTrendAlignment } from "./trend";
import { analyzeMarketStructure } from "./structure";
import { findBestBreakout } from "./breakout";
import { findPullbackEntry } from "./pullback";
import { analyzeMomentum } from "./momentum";
import { analyzeLiquidity } from "./liquidity";
import { analyzeMarketRegime, analyzeRelativeStrength, analyzeVolumeParticipation } from "./context";
import { scoreSetup } from "./scoring";
import { detectChartPattern } from "./chartPatterns";
import { buildExecutionPlan } from "./risk";
import { atr, last, round } from "./indicators";

export interface GradeBucket {
  grade: string;
  trades: number;
  winRate: number;
  avgR: number;
  expectancyR: number;
  targetHitRate: number;
  avgFwdRetPct: number;
}

export interface ScoreBin {
  label: string;
  trades: number;
  winRate: number;
  avgR: number;
}

export interface ValidateResult {
  symbols: number;
  horizonBars: number;
  totalSignals: number;
  byGrade: GradeBucket[];
  byScoreBin: ScoreBin[];
  overall: { trades: number; winRate: number; expectancyR: number; targetHitRate: number };
  monotonicGrade: boolean; // expectancy rises A+ >= A >= B >= C ?
  spearman: number; // rank corr between score and realized R (−1..1)
  notes: string[];
}

interface RawTrade {
  grade: string;
  score: number;
  r: number;
  fwdRetPct: number;
  win: boolean;
  hitTarget: boolean;
}

const GRADES = ["A+", "A", "B", "C"];

/** Build a setup at the close of the sliced daily frame — mirrors runScan's per-symbol block. */
function analyzeAt(
  frame: Bar[],
  benchSlice: Bar[],
  market: Market,
  benchSym: string,
  profile: ScoringProfile,
  family: "breakout",
): { setup: SetupSignal; entry: number; stop: number; target: number } | null {
  if (frame.length < 60) return null;
  const frameMap: Record<string, Bar[]> = { "1d": frame };
  const trend = analyzeTrendAlignment(frameMap, { adxThreshold: 20 });
  const structure = analyzeMarketStructure(frame, { ticker: "X", market, timeframe: "1d" });
  const volume = analyzeVolumeParticipation(frame, structure.keyLevels.bos_level ?? null);
  const momentum = analyzeMomentum(frame);
  const liquidity = analyzeLiquidity(frame, market);
  if (!liquidity.tradable) return null;
  const relativeStrength = analyzeRelativeStrength(frame, benchSlice, { benchmarkSymbol: benchSym, sectorBars: null, sectorSymbol: null });
  const breakout = findBestBreakout(frame, { market, trend, structure, relativeStrengthScore: relativeStrength.score, buffer: 0.5, lookback: 40, eventDays: null });
  if (!breakout.isValid) return null; // only count actual fired signals
  const pullback = findPullbackEntry(frame, breakout, structure);
  const regime = analyzeMarketRegime(benchSlice, { market, benchmarkSymbol: benchSym });

  const setup: SetupSignal = {
    ticker: "X",
    market,
    exchange: "",
    country: market,
    sector: "",
    timeframe: "1d",
    setupFamily: family,
    direction: trend.direction !== "neutral" ? trend.direction : "bullish",
    trend,
    structure,
    breakout,
    pullback: null,
    volume,
    momentum,
    liquidity,
    relativeStrength,
    regime,
    chartPattern: detectChartPattern(frame),
    score: 0,
    grade: "Reject",
    breakdown: {},
    reasonsFor: [],
    reasonsAgainst: [],
    executionPlan: null,
    riskWarnings: [],
    eventRiskDays: null,
    atrPercentile: 0,
  };
  setup.executionPlan = buildExecutionPlan(setup, { capitalBase: CONFIG.runtime.capitalBase, riskPerTradePct: CONFIG.runtime.riskPerTradePct });
  const scored = scoreSetup(setup, profile);
  setup.score = scored.score;
  setup.grade = scored.grade;
  void pullback;
  return { setup, entry: setup.executionPlan.entry, stop: setup.executionPlan.stop, target: setup.executionPlan.target2r };
}

/** Spearman rank correlation between two equal-length arrays. */
function spearman(a: number[], b: number[]): number {
  const n = a.length;
  if (n < 3) return 0;
  const rank = (xs: number[]) => {
    const idx = xs.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(n).fill(0);
    for (let k = 0; k < n; k++) r[idx[k][1]] = k + 1;
    return r;
  };
  const ra = rank(a);
  const rb = rank(b);
  let d2 = 0;
  for (let i = 0; i < n; i++) d2 += (ra[i] - rb[i]) ** 2;
  return round(1 - (6 * d2) / (n * (n * n - 1)), 3);
}

export function validateEngine(
  items: { symbol: string; bars: Bar[] }[],
  bench: Bar[],
  market: Market,
  opts: { horizonBars?: number; profileOverride?: ScoringProfile } = {},
): ValidateResult {
  const horizon = opts.horizonBars ?? 10;
  const profile = opts.profileOverride ?? CONFIG.scoringProfiles.bullish_breakout;
  const benchSym = CONFIG.benchmarkMap[market]?.broad ?? (market === "NSE" ? "^NSEI" : "SPY");
  const trades: RawTrade[] = [];
  const notes: string[] = [];
  let evaluated = 0;

  for (const { bars } of items) {
    if (bars.length < 300) continue;
    // Align benchmark by time so RS/regime never see the future.
    const benchByTime = bench;
    let t = 252;
    while (t < bars.length - horizon - 2) {
      const frame = bars.slice(0, t + 1);
      const cutoff = bars[t].time;
      const benchSlice = benchByTime.filter((b) => b.time <= cutoff);
      const res = analyzeAt(frame, benchSlice, market, benchSym, profile, "breakout");
      if (!res || res.setup.grade === "Reject") {
        t++;
        continue;
      }
      evaluated++;
      const entryIdx = t + 1;
      const entry = bars[entryIdx].open;
      const a = Number.isFinite(last(atr(frame, 14))) ? last(atr(frame, 14)) : entry * 0.03;
      const stop = res.stop < entry ? res.stop : entry - a * 1.5;
      const target = res.target > entry ? res.target : entry + (entry - stop) * 2;
      const lastIdx = Math.min(bars.length - 1, entryIdx + horizon);
      let exitPrice = bars[lastIdx].close;
      let hitTarget = false;
      let exitIdx = lastIdx;
      for (let p = entryIdx; p <= lastIdx; p++) {
        if (bars[p].low <= stop) { exitPrice = stop; exitIdx = p; break; }
        if (bars[p].high >= target) { exitPrice = target; hitTarget = true; exitIdx = p; break; }
      }
      const r = (exitPrice - entry) / Math.max(entry - stop, 1e-9);
      trades.push({
        grade: res.setup.grade,
        score: res.setup.score,
        r,
        fwdRetPct: (exitPrice / entry - 1) * 100,
        win: exitPrice > entry,
        hitTarget,
      });
      t = exitIdx + 1; // no overlapping trades per symbol
    }
  }

  const bucket = (rows: RawTrade[], grade: string): GradeBucket => {
    const n = rows.length;
    return {
      grade,
      trades: n,
      winRate: n ? round((rows.filter((x) => x.win).length / n) * 100, 1) : 0,
      avgR: n ? round(rows.reduce((s, x) => s + x.r, 0) / n, 3) : 0,
      expectancyR: n ? round(rows.reduce((s, x) => s + x.r, 0) / n, 3) : 0,
      targetHitRate: n ? round((rows.filter((x) => x.hitTarget).length / n) * 100, 1) : 0,
      avgFwdRetPct: n ? round(rows.reduce((s, x) => s + x.fwdRetPct, 0) / n, 2) : 0,
    };
  };

  const byGrade = GRADES.map((g) => bucket(trades.filter((x) => x.grade === g), g)).filter((b) => b.trades > 0);

  const bins = [
    { label: "<70", lo: -Infinity, hi: 70 },
    { label: "70–80", lo: 70, hi: 80 },
    { label: "80–85", lo: 80, hi: 85 },
    { label: "85–90", lo: 85, hi: 90 },
    { label: "90+", lo: 90, hi: Infinity },
  ];
  const byScoreBin: ScoreBin[] = bins
    .map((bn) => {
      const rows = trades.filter((x) => x.score >= bn.lo && x.score < bn.hi);
      return { label: bn.label, trades: rows.length, winRate: rows.length ? round((rows.filter((x) => x.win).length / rows.length) * 100, 1) : 0, avgR: rows.length ? round(rows.reduce((s, x) => s + x.r, 0) / rows.length, 3) : 0 };
    })
    .filter((b) => b.trades > 0);

  // Monotonic check: expectancy should not increase as grade weakens (A+ >= A >= B >= C).
  const present = byGrade.map((b) => b.expectancyR);
  let monotonic = true;
  for (let i = 1; i < present.length; i++) if (present[i] > present[i - 1] + 0.05) monotonic = false;

  const overallN = trades.length;
  if (!overallN) notes.push("No fired breakout setups in the window — widen the sample or history.");

  return {
    symbols: items.length,
    horizonBars: horizon,
    totalSignals: evaluated,
    byGrade,
    byScoreBin,
    overall: {
      trades: overallN,
      winRate: overallN ? round((trades.filter((x) => x.win).length / overallN) * 100, 1) : 0,
      expectancyR: overallN ? round(trades.reduce((s, x) => s + x.r, 0) / overallN, 3) : 0,
      targetHitRate: overallN ? round((trades.filter((x) => x.hitTarget).length / overallN) * 100, 1) : 0,
    },
    monotonicGrade: monotonic,
    spearman: spearman(trades.map((x) => x.score), trades.map((x) => x.r)),
    notes,
  };
}
