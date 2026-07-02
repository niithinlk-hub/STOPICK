/**
 * Risk / execution planner. Ported from STOPICK risk/planner.py: fixed-risk
 * position sizing, structure + ATR stop, and a 1R/2R/3R target ladder.
 */
import type { ExecutionPlan, SetupSignal } from "./types";
import { round } from "./indicators";

export function buildExecutionPlan(
  setup: SetupSignal,
  opts: {
    capitalBase: number;
    riskPerTradePct: number;
    portfolioExposurePct?: number;
    correlationPenaltyPct?: number;
    /** Latest ATR(14) of the scan frame. When provided, stop depth is capped at
     * maxStopAtrMult×ATR — structural base lows are often 4–6 ATR deep, which crushes
     * realized R (walk-forward validated: capping raised expectancy in every +EV cell). */
    atrValue?: number;
    maxStopAtrMult?: number;
  },
): ExecutionPlan {
  const portfolioExposurePct = opts.portfolioExposurePct ?? 0;
  const correlationPenaltyPct = opts.correlationPenaltyPct ?? 0;

  let entry: number;
  let stop: number;
  if (setup.pullback && setup.pullback.isValid && setup.pullback.confirmationTrigger) {
    entry = setup.pullback.confirmationTrigger;
    stop = setup.pullback.stopZone ? setup.pullback.stopZone[0] : setup.breakout.invalidationLevel ?? entry * 0.97;
  } else {
    entry = setup.breakout.bufferedLevel ?? setup.breakout.currentPrice;
    stop = setup.breakout.invalidationLevel ?? entry * 0.96;
  }

  const atrStop = Math.min(stop, entry - Math.max(Math.abs(entry - stop), 0.01));
  const structureStop = setup.structure.invalidationLevel ?? stop;
  let finalStop = Math.min(atrStop, structureStop);
  // Cap stop depth at k×ATR below the realistic fill (only binds when the structural stop
  // is deeper). Reference is max(entry, currentPrice): the plan entry is the buffered
  // trigger level, which a fresh breakout has already cleared — real fills land at or
  // above the current price, and the walk-forward that validated the cap entered there.
  if (opts.atrValue !== undefined && Number.isFinite(opts.atrValue) && opts.atrValue > 0) {
    const fillRef = Math.max(entry, setup.breakout.currentPrice);
    finalStop = Math.max(finalStop, fillRef - opts.atrValue * (opts.maxStopAtrMult ?? 3));
  }
  const riskPerShare = Math.max(entry - finalStop, 0.01);
  const capitalRisk = opts.capitalBase * (opts.riskPerTradePct / 100);
  const rawPosition = capitalRisk / riskPerShare;
  const adjustedPosition =
    rawPosition * Math.max(0.1, 1 - portfolioExposurePct / 100 - correlationPenaltyPct / 100);

  const target1r = entry + riskPerShare;
  const target2r = entry + riskPerShare * 2;
  const target3r = entry + riskPerShare * 3;
  const positionShares = Math.max(Math.floor(adjustedPosition), 0);

  const warnings: string[] = [];
  if (setup.eventRiskDays !== null && setup.eventRiskDays <= 7) {
    warnings.push("Event risk is near. Reduce size or skip if the catalyst window matters.");
  }
  if (setup.volume && setup.volume.volumeRatio < 1.3) {
    warnings.push("Participation is not ideal for a high-conviction breakout.");
  }
  if (setup.breakout.distancePct !== null && setup.breakout.distancePct > 6) {
    warnings.push("Price is getting extended away from the breakout reference.");
  }
  if (setup.breakout.extensionAtr !== null && setup.breakout.extensionAtr > 4) {
    warnings.push("Price is stretched well above its 20-EMA — expect mean-reversion risk.");
  }

  return {
    entry: round(entry, 4),
    stop: round(finalStop, 4),
    atrStop: round(atrStop, 4),
    structureStop: round(structureStop, 4),
    target1r: round(target1r, 4),
    target2r: round(target2r, 4),
    target3r: round(target3r, 4),
    trailReference: round(Math.max(target1r, setup.breakout.currentPrice), 4),
    positionSizeShares: positionShares,
    positionValue: round(positionShares * entry, 2),
    capitalRisk: round(capitalRisk, 2),
    rrRatio: round((target2r - entry) / Math.max(entry - finalStop, 1e-9), 2),
    warnings,
  };
}
