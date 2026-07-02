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
    // ₹50 lakh/day hard floor — excludes only untradeable junk. Genuine ₹1000cr+
    // midcaps trade ₹1–4cr/day and must qualify; the liquidity *score* (log-scaled)
    // ranks thinner names lower without removing them. Override with MIN_TURNOVER_INR.
    minTurnoverInr: 5_000_000,
    // Max stop depth in ATR units. Structural stops (base low) are often 4–6 ATR deep,
    // which crushes realized R. Walk-forward (2026-06-26, 4 market×tier cells): capping at
    // 3×ATR raised expectancy in every +EV cell (US t2 +0.082→+0.146, NSE t2 +0.018→+0.075,
    // US t1 +0.006→+0.065) and roughly tripled A-grade expectancy. 2.5 tested slightly worse.
    maxStopAtrMult: 3,
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
  // Weights re-calibrated 2026-06-26 from a walk-forward (Algo Check) over NSE + US large
  // caps (~400 trades, 2 markets × 2 horizons). Relative strength was the ONLY component
  // whose score robustly ranked forward return in BOTH markets, so it is up-weighted (9→16);
  // funded by trimming the measured-flat trend stack (trend_alignment, structure_quality,
  // mtf_agreement, htf_headroom — all ≈0 correlation). A full 16-component refit was tested
  // and REJECTED (it overfit the small per-component noise and did not generalize). Grade
  // thresholds unchanged — only the ranking was re-fit, not the cutoffs.
  scoringProfiles: {
    bullish_breakout: {
      weights: {
        trend_alignment: 9,
        structure_quality: 8,
        breakout_quality: 14,
        pullback_quality: 4,
        volume_confirmation: 11,
        momentum: 9,
        volatility_regime: 6,
        relative_strength: 16,
        liquidity: 12,
        htf_headroom: 4,
        rr_ratio: 7,
        market_regime: 5,
        mtf_agreement: 4,
        follow_through: 6,
        event_risk: 4,
        index_alignment: 6,
      },
      gradeThresholds: { "A+": 90, A: 85, B: 75, C: 65 },
    },
    bullish_pullback: {
      weights: {
        trend_alignment: 11,
        structure_quality: 11,
        breakout_quality: 8,
        pullback_quality: 14,
        volume_confirmation: 9,
        momentum: 8,
        volatility_regime: 6,
        relative_strength: 16,
        liquidity: 12,
        htf_headroom: 4,
        rr_ratio: 7,
        market_regime: 5,
        mtf_agreement: 4,
        follow_through: 4,
        event_risk: 4,
        index_alignment: 6,
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
