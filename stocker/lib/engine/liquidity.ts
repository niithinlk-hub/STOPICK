/**
 * Liquidity / tradability gate. STOCKER accuracy addition — illiquid names
 * produce unreliable breakout signals and are hard to execute, so turnover is
 * scored and (optionally) used to reject names below a floor.
 */
import type { Bar, LiquiditySignal, Market } from "./types";
import { bounded, mean, round } from "./indicators";
import { CONFIG } from "../config";

export function analyzeLiquidity(bars: Bar[], market: Market): LiquiditySignal {
  const recent = bars.slice(-20);
  const avgVolume20 = recent.length ? mean(recent.map((b) => b.volume)) : 0;
  const avgTurnover = recent.length ? mean(recent.map((b) => b.close * b.volume)) : 0;
  const floor = market === "NSE" ? CONFIG.runtime.minTurnoverInr : CONFIG.runtime.minTurnoverUsd;
  const tradable = avgTurnover >= floor;
  // Log-scaled score: floor → ~50, 10x floor → ~100.
  const ratio = floor > 0 ? avgTurnover / floor : 1;
  const score = round(bounded(0.5 + Math.log10(Math.max(ratio, 1e-6)) * 0.5) * 100, 2);
  return {
    avgTurnover: round(avgTurnover, 2),
    avgVolume20: round(avgVolume20, 2),
    tradable,
    score,
    explanation: tradable
      ? `Average 20-bar turnover clears the ${market} liquidity floor.`
      : `Average 20-bar turnover is below the ${market} liquidity floor — execution risk is elevated.`,
  };
}
