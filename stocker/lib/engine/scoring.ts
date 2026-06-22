/**
 * Deterministic, explainable setup scorer. Ported from STOPICK scoring/engine.py
 * and extended with momentum, liquidity, multi-timeframe agreement and
 * follow-through components, plus an over-extension penalty. Output stays 0–100
 * with the same A+/A/B/C/Reject grade ladder so it remains explainable.
 */
import type { Grade, ScoreResult, ScoringProfile, SetupSignal } from "./types";
import { bounded, round } from "./indicators";

export function scoreSetup(setup: SetupSignal, profile: ScoringProfile): ScoreResult {
  const { breakout, pullback, volume, momentum, liquidity, relativeStrength: rs, regime, trend, structure } = setup;

  let rrEstimate = 0;
  if (setup.executionPlan) {
    const { target2r, entry, stop } = setup.executionPlan;
    rrEstimate = (target2r - entry) / Math.max(Math.abs(entry - stop), 1e-9);
  }

  // Unknown earnings proximity (eventRiskDays === null on every scan > 10 symbols) is NOT
  // safe — treat it as neutral 0.7, not a free 1.0, so unscanned earnings risk can't inflate a grade.
  const eventScore = setup.eventRiskDays === null ? 0.7 : setup.eventRiskDays > 7 ? 1 : 0.35;
  const volPenalty = (volume?.penaltyFlags?.length ?? 0) * 0.15;
  const pullbackComponent = pullback === null ? 0.5 : pullback.isValid ? 1 : 0.2;
  const overhead = breakout.overheadResistancePct;
  const headroom = overhead === null ? 1 : overhead > 8 ? 0.75 : overhead > 4 ? 0.45 : 0.1;

  // Follow-through with an over-extension penalty (entries far above EMA20 in ATR
  // units are more likely to fail / mean-revert).
  let followThrough = breakout.followThroughScore / 100;
  if (breakout.extensionAtr !== null && breakout.extensionAtr > 4) {
    followThrough *= bounded(1 - (breakout.extensionAtr - 4) / 6, 0.2, 1);
  }

  const volumeComponent = volume
    ? (volume.volumeRatio - 1) / 1.5 * 0.5 +
      Number(volume.obvConfirmation) * 0.18 +
      Number(volume.vwapAlignment) * 0.12 +
      Number(volume.aboveAnchoredVwap) * 0.1 +
      Number(volume.dryUpBeforeExpansion) * 0.1
    : 0;

  const components: Record<string, number> = {
    trend_alignment: bounded((trend.strengthScore / 100 + trend.alignmentConfidence / 100) / 2),
    structure_quality: bounded(
      Number(structure.bos) * 0.45 +
        Number(!structure.fvgMitigated && structure.fvgZone !== null) * 0.25 +
        Number(structure.liquiditySweep) * 0.15 +
        Number(!structure.choch) * 0.15,
    ),
    breakout_quality: bounded(
      Number(breakout.isValid) * 0.4 +
        (breakout.candleQuality / 100) * 0.25 +
        (breakout.tightnessScore / 100) * 0.2 +
        bounded((breakout.volumeExpansion - 1) / 1.5) * 0.15,
    ),
    pullback_quality: bounded(pullbackComponent),
    volume_confirmation: bounded(volumeComponent - volPenalty),
    momentum: bounded((momentum?.score ?? 50) / 100),
    // Per-symbol volatility: reward a breakout from a quiet/normal base, penalise one
    // firing when the stock's own ATR is already in a blow-off percentile. (Previously this
    // read the INDEX regime and ignored the symbol's atrPercentile entirely.)
    volatility_regime: setup.atrPercentile <= 70 ? 0.85 : setup.atrPercentile <= 90 ? 0.55 : 0.3,
    relative_strength: bounded((rs?.score ?? 0) / 100),
    liquidity: bounded((liquidity?.score ?? 50) / 100),
    htf_headroom: headroom,
    rr_ratio: bounded(rrEstimate / 3),
    market_regime: regime && regime.direction === setup.direction ? 1 : regime && regime.direction === "neutral" ? 0.55 : 0.15,
    mtf_agreement: bounded(trend.mtfAgreement / 100),
    follow_through: bounded(followThrough),
    event_risk: eventScore,
    index_alignment: regime && regime.direction === trend.direction ? 1 : 0.5,
  };

  const totalWeight = Object.values(profile.weights).reduce((a, b) => a + b, 0);
  let weighted = 0;
  for (const [key, weight] of Object.entries(profile.weights)) {
    weighted += (components[key] ?? 0) * weight;
  }
  const finalScore = totalWeight ? round((weighted / totalWeight) * 100, 2) : 0;

  let grade: Grade = "Reject";
  const ordered = Object.entries(profile.gradeThresholds).sort((a, b) => b[1] - a[1]);
  for (const [label, threshold] of ordered) {
    if (finalScore >= threshold) {
      grade = label as Grade;
      break;
    }
  }

  const breakdown: Record<string, number> = {};
  for (const [key, value] of Object.entries(components)) breakdown[key] = round(value * 100, 2);

  return { score: finalScore, grade, breakdown, weights: profile.weights };
}
