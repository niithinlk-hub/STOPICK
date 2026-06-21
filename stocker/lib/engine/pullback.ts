/**
 * Lower-risk pullback / retest entry detection. Ported from STOPICK
 * signals/pullback.py.
 */
import type { Bar, BreakoutSignal, PullbackSignal, StructureSignal } from "./types";
import { atr, last, mean, round } from "./indicators";

export function findPullbackEntry(
  bars: Bar[],
  breakout: BreakoutSignal,
  structure: StructureSignal,
): PullbackSignal {
  if (!bars.length || breakout.breakoutLevel === null) {
    return {
      isValid: false,
      setupType: "none",
      entryZone: null,
      confirmationTrigger: null,
      stopZone: null,
      rrTargets: {},
      explanation: "Breakout level is unavailable for pullback planning.",
      metrics: {},
    };
  }

  const atrValue = last(atr(bars, 14), 0) || 0;
  const latest = bars[bars.length - 1];
  const recent = bars.slice(-8);
  const breakoutLevel = breakout.breakoutLevel;
  const zoneLow = breakoutLevel - atrValue * 0.35;
  const zoneHigh = breakoutLevel + atrValue * 0.2;
  const priceInZone = zoneLow <= latest.close && latest.close <= zoneHigh;

  const priorVol = mean(recent.slice(0, -1).map((b) => b.volume));
  const dryUp = priorVol > 0 && latest.volume <= priorVol * 0.95;
  const rejection = latest.close > latest.open && latest.low <= breakoutLevel && breakoutLevel <= latest.high;
  const fvgFill = !!(structure.fvgZone && structure.fvgZone[0] <= latest.low && latest.low <= structure.fvgZone[1]);

  const confirmationTrigger = Math.max(...recent.map((b) => b.high));
  const risk = Math.max(confirmationTrigger - zoneLow, atrValue || 0.01);
  const rrTargets = {
    "1R": round(confirmationTrigger + risk, 4),
    "2R": round(confirmationTrigger + risk * 2, 4),
    "3R": round(confirmationTrigger + risk * 3, 4),
  };

  const isValid = priceInZone && (rejection || fvgFill) && dryUp;
  return {
    isValid,
    setupType: "retest_pullback",
    entryZone: [round(zoneLow, 4), round(zoneHigh, 4)],
    confirmationTrigger: round(confirmationTrigger, 4),
    stopZone: [round(zoneLow - atrValue * 0.35, 4), round(zoneLow, 4)],
    rrTargets,
    explanation: isValid
      ? "Price has retested the breakout region with lower participation and a reclaim-style candle."
      : "No lower-risk pullback entry is active right now.",
    metrics: { dry_up: dryUp, rejection, fvg_fill: fvgFill, price_in_zone: priceInZone },
  };
}
