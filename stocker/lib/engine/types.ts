/**
 * STOCKER engine type contract.
 *
 * Ported from STOPICK's pydantic models (signals/models.py) and extended with
 * the accuracy-tuning signals that STOCKER adds: momentum (RSI/MACD), liquidity
 * (turnover), and an explicit multi-timeframe agreement factor.
 *
 * Every engine module and every API route depends on these shapes. Treat this
 * file as the single source of truth — do not redefine these inline elsewhere.
 */

export type Country = "NSE" | "US" | "BOTH";
export type Market = "NSE" | "US";
export type Timeframe = "1d" | "4h" | "1h" | "15m";
export type SetupFamily = "breakout" | "pullback";
export type SetupMode = SetupFamily | "both";
export type Direction = "bullish" | "bearish" | "neutral";
export type Grade = "A+" | "A" | "B" | "C" | "Reject";
export type UniverseSource = "tier_1" | "tier_2" | "tier_3" | "sample" | "manual";

/** Unified OHLCV bar. `time` is epoch seconds (UTC) — matches lightweight-charts. */
export interface Bar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SymbolRecord {
  /** Provider symbol, e.g. "RELIANCE.NS" or "AAPL". */
  symbol: string;
  /** Display symbol with the `.NS` suffix stripped. */
  display: string;
  market: Market;
  exchange: string;
  sector: string;
  marketCapBucket: string;
}

export interface TrendSignal {
  direction: Direction;
  /** 0–100 average of per-timeframe checklist scores. */
  strengthScore: number;
  /** 0–100 fraction of timeframes that agree on direction. */
  alignmentConfidence: number;
  timeframeScores: Record<string, number>;
  /** 0–100 — STOCKER addition: how aligned the stacked timeframes are. */
  mtfAgreement: number;
  metrics: Record<string, number | string | null>;
}

export interface StructureSignal {
  ticker: string;
  market: Market;
  timeframe: Timeframe;
  direction: Direction;
  structureType: string;
  keyLevels: Record<string, number | null>;
  retestZone: [number, number] | null;
  invalidationLevel: number | null;
  bos: boolean;
  choch: boolean;
  liquiditySweep: boolean;
  equalHighs: boolean;
  equalLows: boolean;
  inducement: boolean;
  orderBlockZone: [number, number] | null;
  fvgZone: [number, number] | null;
  fvgMitigated: boolean;
  explanation: string;
}

export interface BreakoutSignal {
  isValid: boolean;
  patternName: string;
  direction: Direction;
  breakoutLevel: number | null;
  bufferedLevel: number | null;
  currentPrice: number;
  distancePct: number | null;
  /** 0–100 close-near-high / body quality of the breakout bar. */
  candleQuality: number;
  /** 0–100 pre-breakout base tightness. */
  tightnessScore: number;
  volumeExpansion: number;
  overheadResistancePct: number | null;
  invalidationLevel: number | null;
  /** STOCKER addition: 0–100 confidence the breakout is not a false break. */
  followThroughScore: number;
  /** STOCKER addition: distance of price above EMA20 in ATR units. */
  extensionAtr: number | null;
  explanation: string;
  metrics: Record<string, unknown>;
}

export interface PullbackSignal {
  isValid: boolean;
  setupType: string;
  entryZone: [number, number] | null;
  confirmationTrigger: number | null;
  stopZone: [number, number] | null;
  rrTargets: Record<string, number>;
  explanation: string;
  metrics: Record<string, unknown>;
}

export interface VolumeSignal {
  volumeRatio: number;
  relativeVolume: number;
  obvConfirmation: boolean;
  vwapAlignment: boolean;
  anchoredVwap: number | null;
  /** STOCKER addition: price is above the breakout-anchored VWAP. */
  aboveAnchoredVwap: boolean;
  dryUpBeforeExpansion: boolean;
  /** STOCKER addition: 20-bar OBV/A-D slope sign, -1..1. */
  accumulationTrend: number;
  penaltyFlags: string[];
}

/** STOCKER addition — momentum confirmation (RSI + MACD). */
export interface MomentumSignal {
  rsi: number;
  /** "oversold" | "healthy" | "strong" | "overbought" */
  rsiState: string;
  macd: number;
  macdSignal: number;
  macdHistogram: number;
  /** true when histogram has just turned/stayed positive. */
  macdBullish: boolean;
  /** 0–100 composite momentum quality used by the scorer. */
  score: number;
  explanation: string;
}

/** STOCKER addition — liquidity / tradability gate. */
export interface LiquiditySignal {
  /** 20-bar average traded value in the instrument's currency. */
  avgTurnover: number;
  avgVolume20: number;
  /** true when turnover clears the configured minimum. */
  tradable: boolean;
  /** 0–100 normalized liquidity score. */
  score: number;
  explanation: string;
}

export interface RelativeStrengthSignal {
  benchmarkSymbol: string;
  sectorBenchmarkSymbol: string | null;
  score: number;
  trendPersistence: number;
  smoothness: number;
  oneWeekAlpha: number;
  oneMonthAlpha: number;
  threeMonthAlpha: number;
  /** STOCKER addition: alpha vs the sector benchmark when available. */
  sectorAlpha: number | null;
  explanation: string;
}

export interface RegimeSignal {
  market: Market;
  benchmarkSymbol: string;
  direction: Direction;
  trendStrength: number;
  volatilityState: string;
  breadthLikeProxy: number;
  explanation: string;
}

/** STOCKER addition — classical chart-pattern detection (flags/triangles/cup-handle). */
export interface ChartPattern {
  key: string;
  name: string;
  direction: Direction;
  /** 0–100 geometric confidence. Descriptive only — not part of the composite score. */
  confidence: number;
  description: string;
}

export interface ExecutionPlan {
  entry: number;
  stop: number;
  atrStop: number;
  structureStop: number;
  target1r: number;
  target2r: number;
  target3r: number;
  trailReference: number;
  positionSizeShares: number;
  positionValue: number;
  capitalRisk: number;
  rrRatio: number;
  warnings: string[];
}

export interface ScoreResult {
  score: number;
  grade: Grade;
  /** component key -> 0–100 contribution value */
  breakdown: Record<string, number>;
  /** component key -> weight used */
  weights: Record<string, number>;
}

export interface SetupSignal {
  ticker: string;
  market: Market;
  exchange: string;
  country: Market;
  sector: string;
  timeframe: Timeframe;
  setupFamily: SetupFamily;
  direction: Direction;
  trend: TrendSignal;
  structure: StructureSignal;
  breakout: BreakoutSignal;
  pullback: PullbackSignal | null;
  volume: VolumeSignal | null;
  momentum: MomentumSignal | null;
  liquidity: LiquiditySignal | null;
  relativeStrength: RelativeStrengthSignal | null;
  regime: RegimeSignal | null;
  chartPattern: ChartPattern | null;
  score: number;
  grade: Grade;
  breakdown: Record<string, number>;
  reasonsFor: string[];
  reasonsAgainst: string[];
  executionPlan: ExecutionPlan | null;
  riskWarnings: string[];
  eventRiskDays: number | null;
  atrPercentile: number;
}

/* ----------------------------- Configuration ----------------------------- */

export interface ScoringProfile {
  weights: Record<string, number>;
  gradeThresholds: Record<string, number>;
}

export interface RuntimeConfig {
  capitalBase: number;
  riskPerTradePct: number;
  minScoreDefault: number;
  maxSymbolsPerScan: number;
  slippageBps: number;
  brokerageBps: number;
  taxesBps: number;
  /** STOCKER addition — minimum 20d turnover to be considered tradable. */
  minTurnoverUsd: number;
  minTurnoverInr: number;
  /** Max stop depth in ATR units — structural stops deeper than this are capped (walk-forward validated). */
  maxStopAtrMult: number;
}

export interface AppConfig {
  runtime: RuntimeConfig;
  benchmarkMap: Record<Market, Record<string, string>>;
  /** Map of sector name -> benchmark symbol, per market (sector RS routing). */
  sectorBenchmarkMap: Record<Market, Record<string, string>>;
  scoringProfiles: Record<string, ScoringProfile>;
}

/* ------------------------------- API shapes ------------------------------ */

export interface ScanParams {
  country: Country;
  source: UniverseSource;
  timeframe: Timeframe;
  setupMode: SetupMode;
  minScore: number;
  manualSymbols?: string;
  limit?: number;
  /** Inject a live forming daily candle from Dhan quotes (pre-close runs). NSE only. */
  live?: boolean;
  /** Also collect pre-breakout "coiling" candidates into ScanResponse.watch. */
  includeWatch?: boolean;
}

export interface ScanRow {
  ticker: string;
  market: Market;
  sector: string;
  timeframe: Timeframe;
  setupFamily: SetupFamily;
  pattern: string;
  chartPattern: string | null;
  chartPatternConfidence: number | null;
  score: number;
  grade: Grade;
  direction: Direction;
  trendStrength: number;
  breakoutLevel: number | null;
  currentPrice: number;
  distancePct: number | null;
  rsScore: number | null;
  rsiState: string | null;
  volumeRatio: number | null;
  atrPercentile: number;
  rrRatio: number | null;
  entry: number | null;
  stop: number | null;
  target2r: number | null;
  eventRiskDays: number | null;
  whyQualified: string;
}

/** A pre-breakout ("coiling") candidate — tight base just under its trigger, not yet broken. */
export interface WatchRow {
  ticker: string;
  market: Market;
  sector: string;
  pattern: string;
  trigger: number;       // breakout level to clear
  currentPrice: number;
  distancePct: number;   // negative — % below the trigger
  tightness: number;     // 0..100, higher = tighter coil
  rsScore: number | null;
  readiness: number;     // 0..100 composite (proximity + tightness + RS + dry-up)
}

export interface ScanResponse {
  rows: ScanRow[];
  setups: SetupSignal[];
  regimes: Record<string, RegimeSignal>;
  scannedSymbols: number;
  successfulSymbols: number;
  qualifiedSymbols: number;
  failures: Record<string, string>;
  notes: string[];
  generatedAt: number;
  elapsedMs: number;
  /** Pre-breakout watch list (populated when ScanParams.includeWatch). */
  watch?: WatchRow[];
}

export interface SymbolDetailResponse {
  ticker: string;
  market: Market;
  timeframe: Timeframe;
  bars: Bar[];
  setup: SetupSignal | null;
}

/* ----------------------- RSI Confluence strategy ------------------------ */

export type ConfluenceSide = "long" | "short" | "none";

/** One symbol's read for the Stochastic + RSI(50) + MACD-cross confluence screen. */
export interface ConfluenceRow {
  ticker: string;
  market: Market;
  timeframe: Timeframe;
  side: ConfluenceSide;
  signal: boolean;
  confidence: number; // 0–100 alignment strength
  price: number;
  stochK: number;
  stochD: number;
  stochState: string; // oversold | overbought | rising | falling
  rsi: number;
  rsiAbove50: boolean;
  macdHist: number;
  macdCrossUp: boolean;
  macdCrossDown: boolean;
  entry: number | null;
  stop: number | null;
  target: number | null; // 1.5R
  rr: number;
  stopPct: number | null; // stop distance as % of entry
  reasons: string[];
}

export interface ConfluenceResponse {
  rows: ConfluenceRow[];
  scannedSymbols: number;
  successfulSymbols: number;
  signalCount: number;
  failures: Record<string, string>;
  notes: string[];
  timeframe: Timeframe;
  mode: string;
  generatedAt: number;
  elapsedMs: number;
}
