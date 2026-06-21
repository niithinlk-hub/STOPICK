import type { AppConfig, Grade } from "./engine/types";

/**
 * STOCKER configuration. Ported from STOPICK's config/config.yaml and extended
 * with the new accuracy components (momentum, liquidity, mtf_agreement,
 * follow_through) and sector relative-strength routing.
 */
export const CONFIG: AppConfig = {
  runtime: {
    capitalBase: 1_000_000,
    riskPerTradePct: 0.5,
    minScoreDefault: 75,
    maxSymbolsPerScan: 500,
    slippageBps: 5,
    brokerageBps: 3,
    taxesBps: 2,
    minTurnoverUsd: 5_000_000,
    minTurnoverInr: 50_000_000,
  },
  benchmarkMap: {
    NSE: { broad: "^NSEI", broad_alt: "^CRSLDX", bank: "^NSEBANK", tech: "^CNXIT" },
    US: { broad: "SPY", growth: "QQQ", semis: "SMH", financials: "XLF" },
  },
  // Sector RS routing (README TODO #1). Falls back to the broad benchmark when a
  // sector is not mapped here.
  sectorBenchmarkMap: {
    NSE: {
      Bank: "^NSEBANK",
      Financials: "^NSEBANK",
      Technology: "^CNXIT",
      IT: "^CNXIT",
    },
    US: {
      Technology: "QQQ",
      Semiconductors: "SMH",
      Financials: "XLF",
      Financial: "XLF",
    },
  },
  scoringProfiles: {
    bullish_breakout: {
      weights: {
        trend_alignment: 13,
        structure_quality: 10,
        breakout_quality: 14,
        pullback_quality: 4,
        volume_confirmation: 11,
        momentum: 9,
        volatility_regime: 6,
        relative_strength: 9,
        liquidity: 5,
        htf_headroom: 6,
        rr_ratio: 7,
        market_regime: 5,
        mtf_agreement: 6,
        follow_through: 6,
        event_risk: 4,
        index_alignment: 4,
      },
      gradeThresholds: { "A+": 90, A: 85, B: 75, C: 65 },
    },
    bullish_pullback: {
      weights: {
        trend_alignment: 13,
        structure_quality: 12,
        breakout_quality: 8,
        pullback_quality: 14,
        volume_confirmation: 9,
        momentum: 8,
        volatility_regime: 6,
        relative_strength: 9,
        liquidity: 5,
        htf_headroom: 6,
        rr_ratio: 7,
        market_regime: 5,
        mtf_agreement: 6,
        follow_through: 4,
        event_risk: 4,
        index_alignment: 4,
      },
      gradeThresholds: { "A+": 90, A: 85, B: 75, C: 65 },
    },
  },
};

export const GRADE_ORDER: Grade[] = ["A+", "A", "B", "C", "Reject"];

/** Human-readable labels for score-breakdown component keys. */
export const COMPONENT_LABELS: Record<string, string> = {
  trend_alignment: "Trend Alignment",
  structure_quality: "Market Structure",
  breakout_quality: "Breakout Quality",
  pullback_quality: "Pullback Quality",
  volume_confirmation: "Volume",
  momentum: "Momentum (RSI/MACD)",
  volatility_regime: "Volatility Regime",
  relative_strength: "Relative Strength",
  liquidity: "Liquidity",
  htf_headroom: "Headroom",
  rr_ratio: "Risk / Reward",
  market_regime: "Market Regime",
  mtf_agreement: "Multi-Timeframe",
  follow_through: "Follow-Through",
  event_risk: "Event Risk",
  index_alignment: "Index Alignment",
};

export const SUPPORTED_INTERVALS = ["1d", "4h", "1h", "15m"] as const;
