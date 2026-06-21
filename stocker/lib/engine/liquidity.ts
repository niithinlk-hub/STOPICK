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
  const envInr = Number(process.env.MIN_TURNOVER_INR);
  const floor =
    market === "NSE"
      ? Number.isFinite(envInr) && envInr > 0
        ? envInr
        : CONFIG.runtime.minTurnoverInr
      : CONFIG.runtime.minTurnoverUsd;
  const tradable = avgTurnover >= floor;
  // Score is graded against a "healthy" daily turnover reference (₹5cr NSE / $5M US),
  // not the hard floor — so dropping the floor admits thin names but still ranks them
  // low. ref → ~50, 10x ref → ~100, 0.1x ref → ~0. (US unchanged from prior behavior.)
  const ref = market === "NSE" ? 50_000_000 : CONFIG.runtime.minTurnoverUsd;
  const ratio = ref > 0 ? avgTurnover / ref : 1;
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
