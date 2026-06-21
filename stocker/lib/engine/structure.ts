/**
 * Market-structure analysis (BOS / CHOCH, pivots, FVG, order block, liquidity
 * sweep). Ported from STOPICK signals/structure.py.
 */
import type { Bar, Direction, Market, StructureSignal, Timeframe } from "./types";
import { atr, last } from "./indicators";

function confirmedPivots(bars: Bar[], left = 3, right = 3): { highs: [number, number][]; lows: [number, number][] } {
  const highs: [number, number][] = [];
  const lows: [number, number][] = [];
  for (let idx = left; idx < bars.length - right; idx++) {
    const window = bars.slice(idx - left, idx + right + 1);
    const ch = bars[idx].high;
    const cl = bars[idx].low;
    if (ch >= Math.max(...window.map((b) => b.high))) highs.push([idx, ch]);
    if (cl <= Math.min(...window.map((b) => b.low))) lows.push([idx, cl]);
  }
  return { highs, lows };
}

function latestFvg(bars: Bar[]): { zone: [number, number] | null; mitigated: boolean } {
  let zone: [number, number] | null = null;
  let mitigated = false;
  const lastBar = bars[bars.length - 1];
  for (let idx = 2; idx < bars.length; idx++) {
    const c1 = bars[idx - 2];
    const c3 = bars[idx];
    if (c3.low > c1.high) {
      zone = [c1.high, c3.low];
      mitigated = lastBar.low <= zone[1] && lastBar.high >= zone[0];
    } else if (c3.high < c1.low) {
      zone = [c3.high, c1.low];
      mitigated = lastBar.low <= zone[1] && lastBar.high >= zone[0];
    }
  }
  return { zone, mitigated };
}

export function analyzeMarketStructure(
  bars: Bar[],
  opts: { ticker: string; market: Market; timeframe: Timeframe; pivotLeft?: number; pivotRight?: number; equalTolerancePct?: number },
): StructureSignal {
  const { ticker, market, timeframe } = opts;
  const pivotLeft = opts.pivotLeft ?? 3;
  const pivotRight = opts.pivotRight ?? 3;
  const equalTolerancePct = opts.equalTolerancePct ?? 0.2;

  const base: StructureSignal = {
    ticker,
    market,
    timeframe,
    direction: "neutral",
    structureType: "insufficient_data",
    keyLevels: {},
    retestZone: null,
    invalidationLevel: null,
    bos: false,
    choch: false,
    liquiditySweep: false,
    equalHighs: false,
    equalLows: false,
    inducement: false,
    orderBlockZone: null,
    fvgZone: null,
    fvgMitigated: false,
    explanation: "Not enough candles to confirm market structure.",
  };

  if (bars.length < pivotLeft + pivotRight + 10) return base;

  const { highs, lows } = confirmedPivots(bars, pivotLeft, pivotRight);
  const lastBar = bars[bars.length - 1];
  const latestClose = lastBar.close;
  const atrValue = last(atr(bars, 14), 0) || 0;

  const previousHigh = highs.length >= 2 ? highs[highs.length - 2][1] : null;
  const previousLow = lows.length >= 2 ? lows[lows.length - 2][1] : null;
  const currentSwingHigh = highs.length ? highs[highs.length - 1][1] : null;
  const currentSwingLow = lows.length ? lows[lows.length - 1][1] : null;

  const bullishBos = previousHigh !== null && latestClose > previousHigh;
  const bearishBos = previousLow !== null && latestClose < previousLow;

  let choch = false;
  if (highs.length >= 2 && lows.length >= 2) {
    const hLast = highs[highs.length - 1][1];
    const hPrev = highs[highs.length - 2][1];
    const lLast = lows[lows.length - 1][1];
    const lPrev = lows[lows.length - 2][1];
    choch = (hLast < hPrev && lLast < lPrev && bullishBos) || (hLast > hPrev && lLast > lPrev && bearishBos);
  }

  const equalHighs =
    highs.length >= 2 && previousHigh !== null && previousHigh !== 0
      ? (Math.abs(highs[highs.length - 1][1] - previousHigh) / previousHigh) * 100 <= equalTolerancePct
      : false;
  const equalLows =
    lows.length >= 2 && previousLow !== null && previousLow !== 0
      ? (Math.abs(lows[lows.length - 1][1] - previousLow) / previousLow) * 100 <= equalTolerancePct
      : false;

  const liquiditySweep =
    (previousHigh !== null && lastBar.high > previousHigh && latestClose < previousHigh) ||
    (previousLow !== null && lastBar.low < previousLow && latestClose > previousLow);

  const inducement =
    lows.length >= 3 &&
    previousLow !== null &&
    lows[lows.length - 1][1] > lows[lows.length - 2][1] &&
    lows[lows.length - 2][1] > lows[lows.length - 3][1];

  const direction: Direction = bullishBos ? "bullish" : bearishBos ? "bearish" : "neutral";
  const structureType = bullishBos || bearishBos ? "BOS" : choch ? "CHOCH" : "range";

  let orderBlockZone: [number, number] | null = null;
  if (bullishBos) {
    for (let i = bars.length - 2; i >= 0; i--) {
      if (bars[i].close < bars[i].open) {
        orderBlockZone = [bars[i].low, bars[i].high];
        break;
      }
    }
  } else if (bearishBos) {
    for (let i = bars.length - 2; i >= 0; i--) {
      if (bars[i].close > bars[i].open) {
        orderBlockZone = [bars[i].low, bars[i].high];
        break;
      }
    }
  }

  const { zone: fvgZone, mitigated: fvgMitigated } = latestFvg(bars);
  const bosLevel = bullishBos ? previousHigh : bearishBos ? previousLow : currentSwingHigh ?? currentSwingLow;
  let retestZone: [number, number] | null = null;
  if (bosLevel !== null && bosLevel !== undefined && atrValue) {
    retestZone = [bosLevel - atrValue * 0.25, bosLevel + atrValue * 0.25];
  }

  const explanation = [
    `Latest confirmed structure is ${structureType} with ${direction} bias.`,
    liquiditySweep ? "Liquidity sweep detected." : "No fresh sweep on the latest bar.",
    fvgZone && !fvgMitigated ? "Fresh FVG available." : "No fresh unmitigated FVG.",
  ].join(" ");

  return {
    ticker,
    market,
    timeframe,
    direction,
    structureType,
    keyLevels: {
      previous_high: previousHigh,
      previous_low: previousLow,
      current_swing_high: currentSwingHigh,
      current_swing_low: currentSwingLow,
      bos_level: bosLevel ?? null,
    },
    retestZone,
    invalidationLevel: direction === "bullish" ? currentSwingLow : currentSwingHigh,
    bos: bullishBos || bearishBos,
    choch,
    liquiditySweep,
    equalHighs,
    equalLows,
    inducement,
    orderBlockZone,
    fvgZone,
    fvgMitigated,
    explanation,
  };
}
