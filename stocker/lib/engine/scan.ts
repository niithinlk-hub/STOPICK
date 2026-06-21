/**
 * Scan orchestrator (server-only). Ported from STOPICK stopick_app/workstation.py
 * scan_market: resolve a universe, pull multi-timeframe data, run every signal
 * engine, build + score + size each setup, and return ranked rows.
 */
import "server-only";
import type {
  Bar,
  RegimeSignal,
  ScanParams,
  ScanResponse,
  ScanRow,
  SetupFamily,
  SetupSignal,
  Timeframe,
} from "./types";
import { CONFIG } from "../config";
import { fetchEventDays, fetchOhlcv } from "../data/yahoo";
import { mapWithConcurrency } from "../data/cache";
import { resolveRecords } from "../data/universe";
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
import { atr, last, rollingPercentile, round } from "./indicators";
import type { Market } from "./types";

function timeframesForScan(tf: Timeframe): Timeframe[] {
  if (tf === "1d") return ["1d"];
  if (tf === "4h") return ["1d", "4h"];
  if (tf === "1h") return ["1d", "4h", "1h"];
  return ["1d", "4h", "1h", "15m"];
}

function benchmarkSymbol(market: Market): string {
  return CONFIG.benchmarkMap[market]?.broad ?? (market === "NSE" ? "^NSEI" : "SPY");
}

function sectorBenchmarkSymbol(market: Market, sector: string): string | null {
  return CONFIG.sectorBenchmarkMap[market]?.[sector] ?? null;
}

/**
 * Evaluate a single symbol end-to-end (no min-score filtering). Used by the
 * symbol-detail API to drive the chart + breakdown panel.
 */
export async function analyzeSymbol(
  symbol: string,
  market: Market,
  timeframe: Timeframe,
  family: SetupFamily = "breakout",
): Promise<{ bars: Bar[]; setup: SetupSignal | null }> {
  const display = symbol.endsWith(".NS") ? symbol.slice(0, -3) : symbol;
  const tfs = timeframesForScan(timeframe);
  const frameMap: Record<string, Bar[]> = {};
  for (const tf of tfs) {
    if (tf === "15m" && timeframe !== "15m") continue;
    frameMap[tf] = await fetchOhlcv(symbol, tf, 400);
  }
  const scanFrame = frameMap[timeframe]?.length ? frameMap[timeframe] : frameMap["1d"] ?? [];
  if (!scanFrame.length) return { bars: [], setup: null };

  const benchSym = benchmarkSymbol(market);
  const benchBars = await fetchOhlcv(benchSym, "1d", 400);
  const regime = analyzeMarketRegime(benchBars, { market, benchmarkSymbol: benchSym });

  const trend = analyzeTrendAlignment(frameMap, { adxThreshold: 20 });
  const structure = analyzeMarketStructure(scanFrame, { ticker: display, market, timeframe });
  const volume = analyzeVolumeParticipation(scanFrame, structure.keyLevels.bos_level ?? null);
  const momentum = analyzeMomentum(scanFrame);
  const liquidity = analyzeLiquidity(scanFrame, market);
  const relativeStrength = analyzeRelativeStrength(scanFrame, benchBars, { benchmarkSymbol: benchSym });
  const breakout = findBestBreakout(scanFrame, {
    market,
    trend,
    structure,
    relativeStrengthScore: relativeStrength.score,
  });
  const pullback = findPullbackEntry(scanFrame, breakout, structure);
  const atrPctile = scanFrame.length >= 30 ? round(last(rollingPercentile(atr(scanFrame, 14), 252), 0) * 100, 2) : 0;

  const setup: SetupSignal = {
    ticker: display,
    market,
    exchange: market,
    country: market,
    sector: "Unknown",
    timeframe,
    setupFamily: family,
    direction: trend.direction !== "neutral" ? trend.direction : "bullish",
    trend,
    structure,
    breakout,
    pullback,
    volume,
    momentum,
    liquidity,
    relativeStrength,
    regime,
    chartPattern: detectChartPattern(scanFrame),
    score: 0,
    grade: "Reject",
    breakdown: {},
    reasonsFor: [breakout.explanation, structure.explanation, relativeStrength.explanation, momentum.explanation],
    reasonsAgainst: [...volume.penaltyFlags, ...(liquidity.tradable ? [] : ["below_liquidity_floor"])],
    executionPlan: null,
    riskWarnings: [],
    eventRiskDays: null,
    atrPercentile: atrPctile,
  };
  setup.executionPlan = buildExecutionPlan(setup, {
    capitalBase: CONFIG.runtime.capitalBase,
    riskPerTradePct: CONFIG.runtime.riskPerTradePct,
  });
  setup.riskWarnings = setup.executionPlan.warnings;
  const profile = CONFIG.scoringProfiles[family === "pullback" ? "bullish_pullback" : "bullish_breakout"];
  const { score, grade, breakdown } = scoreSetup(setup, profile);
  setup.score = score;
  setup.grade = grade;
  setup.breakdown = breakdown;

  return { bars: scanFrame, setup };
}

export async function runScan(params: ScanParams): Promise<ScanResponse> {
  const startedAt = Date.now();
  const records = resolveRecords({
    country: params.country,
    source: params.source,
    manualSymbols: params.manualSymbols,
    maxSymbols: Math.min(params.limit ?? CONFIG.runtime.maxSymbolsPerScan, CONFIG.runtime.maxSymbolsPerScan),
  });

  const notes: string[] = [];
  const failures: Record<string, string> = {};
  const tfs = timeframesForScan(params.timeframe);

  // Benchmark frames + regimes (fetched once per market).
  const benchmarkFrames: Record<string, Bar[]> = {};
  const regimes: Record<string, RegimeSignal> = {};
  const markets = [...new Set(records.map((r) => r.market))];
  await Promise.all(
    markets.map(async (market) => {
      const sym = benchmarkSymbol(market);
      const bars = await fetchOhlcv(sym, "1d", 400);
      benchmarkFrames[sym] = bars;
      regimes[market] = analyzeMarketRegime(bars, { market, benchmarkSymbol: sym });
    }),
  );

  // Event-calendar lookup is only worth it on small scans (matches STOPICK).
  const eventDays: Record<string, number | null> = {};
  if (records.length > 10) {
    notes.push("Event-calendar lookup was skipped for this larger scan to keep the scanner responsive.");
  } else {
    await Promise.all(
      records.map(async (r) => {
        eventDays[r.display] = await fetchEventDays(r.symbol);
      }),
    );
  }

  const setups: SetupSignal[] = [];

  await mapWithConcurrency(records, 8, async (record) => {
    try {
      const frameMap: Record<string, Bar[]> = {};
      for (const tf of tfs) {
        if (tf === "15m" && params.timeframe !== "15m") continue;
        frameMap[tf] = await fetchOhlcv(record.symbol, tf, 400);
      }
      let scanFrame = frameMap[params.timeframe];
      if (!scanFrame || !scanFrame.length) scanFrame = frameMap["1d"];
      if (!scanFrame || !scanFrame.length) {
        failures[record.symbol] = "No scan frame data.";
        return;
      }

      const benchSym = benchmarkSymbol(record.market);
      const benchBars = benchmarkFrames[benchSym] ?? [];
      const sectorSym = sectorBenchmarkSymbol(record.market, record.sector);
      const sectorBars = sectorSym ? await fetchOhlcv(sectorSym, "1d", 400) : null;

      const trend = analyzeTrendAlignment(frameMap, { adxThreshold: 20 });
      const structure = analyzeMarketStructure(scanFrame, {
        ticker: record.display,
        market: record.market,
        timeframe: params.timeframe,
      });
      const volume = analyzeVolumeParticipation(scanFrame, structure.keyLevels.bos_level ?? null);
      const momentum = analyzeMomentum(scanFrame);
      const liquidity = analyzeLiquidity(scanFrame, record.market);
      // Hard liquidity gate: below the turnover floor a name is untradeable (slippage
      // destroys the plan), so exclude it from results rather than letting it grade A.
      if (!liquidity.tradable) {
        failures[record.symbol] = "below_liquidity_floor";
        return;
      }
      const relativeStrength = analyzeRelativeStrength(scanFrame, benchBars, {
        benchmarkSymbol: benchSym,
        sectorBars,
        sectorSymbol: sectorSym,
      });
      const breakout = findBestBreakout(scanFrame, {
        market: record.market,
        trend,
        structure,
        relativeStrengthScore: relativeStrength.score,
        buffer: 0.5,
        lookback: 40,
        eventDays: eventDays[record.display] ?? null,
      });
      const pullback = findPullbackEntry(scanFrame, breakout, structure);
      const atrPctile =
        scanFrame.length >= 30 ? round(last(rollingPercentile(atr(scanFrame, 14), 252), 0) * 100, 2) : 0;
      const chartPattern = detectChartPattern(scanFrame);

      const families: SetupFamily[] = params.setupMode === "both" ? ["breakout", "pullback"] : [params.setupMode];
      for (const family of families) {
        if (family === "breakout" && !breakout.isValid) continue;
        if (family === "pullback" && !pullback.isValid) continue;

        const setup: SetupSignal = {
          ticker: record.display,
          market: record.market,
          exchange: record.exchange,
          country: record.market,
          sector: record.sector,
          timeframe: params.timeframe,
          setupFamily: family,
          direction: trend.direction !== "neutral" ? trend.direction : "bullish",
          trend,
          structure,
          breakout,
          pullback: family === "pullback" ? pullback : null,
          volume,
          momentum,
          liquidity,
          relativeStrength,
          regime: regimes[record.market] ?? null,
          chartPattern,
          score: 0,
          grade: "Reject",
          breakdown: {},
          reasonsFor: [
            breakout.explanation,
            structure.explanation,
            relativeStrength.explanation,
            volume.volumeRatio >= 1.5
              ? "Volume participation confirms the move."
              : "Participation is acceptable but not exceptional.",
            momentum.explanation,
          ],
          reasonsAgainst: [...volume.penaltyFlags, ...(liquidity.tradable ? [] : ["below_liquidity_floor"])],
          executionPlan: null,
          riskWarnings: [],
          eventRiskDays: eventDays[record.display] ?? null,
          atrPercentile: atrPctile,
        };

        setup.executionPlan = buildExecutionPlan(setup, {
          capitalBase: CONFIG.runtime.capitalBase,
          riskPerTradePct: CONFIG.runtime.riskPerTradePct,
        });
        setup.riskWarnings = setup.executionPlan.warnings;

        const profile = CONFIG.scoringProfiles[family === "pullback" ? "bullish_pullback" : "bullish_breakout"];
        const { score, grade, breakdown } = scoreSetup(setup, profile);
        setup.score = score;
        setup.grade = grade;
        setup.breakdown = breakdown;

        if (score >= params.minScore) setups.push(setup);
      }
    } catch (err) {
      failures[record.symbol] = err instanceof Error ? err.message : String(err);
    }
  });

  setups.sort((a, b) => b.score - a.score || (b.relativeStrength?.score ?? 0) - (a.relativeStrength?.score ?? 0));

  const rows: ScanRow[] = setups.map((s) => ({
    ticker: s.ticker,
    market: s.market,
    sector: s.sector,
    timeframe: s.timeframe,
    setupFamily: s.setupFamily,
    pattern: s.breakout.patternName,
    chartPattern: s.chartPattern?.name ?? null,
    chartPatternConfidence: s.chartPattern?.confidence ?? null,
    score: s.score,
    grade: s.grade,
    direction: s.direction,
    trendStrength: s.trend.strengthScore,
    breakoutLevel: s.breakout.breakoutLevel,
    currentPrice: s.breakout.currentPrice,
    distancePct: s.breakout.distancePct,
    rsScore: s.relativeStrength?.score ?? null,
    rsiState: s.momentum?.rsiState ?? null,
    volumeRatio: s.volume?.volumeRatio ?? null,
    atrPercentile: s.atrPercentile,
    rrRatio: s.executionPlan?.rrRatio ?? null,
    entry: s.executionPlan?.entry ?? null,
    stop: s.executionPlan?.stop ?? null,
    target2r: s.executionPlan?.target2r ?? null,
    eventRiskDays: s.eventRiskDays,
    whyQualified: s.reasonsFor.slice(0, 3).join(" | "),
  }));

  const successful = records.length - Object.keys(failures).length;
  return {
    rows,
    setups,
    regimes,
    scannedSymbols: records.length,
    successfulSymbols: successful,
    qualifiedSymbols: setups.length,
    failures,
    notes,
    generatedAt: startedAt,
    elapsedMs: Date.now() - startedAt,
  };
}
