/**
 * Classical chart-pattern detection — flags, triangles, and cup-and-handle.
 *
 * Deterministic geometry on OHLC bars (swing pivots + least-squares trendlines),
 * tuned for daily data. Each detector returns a 0–100 confidence; the public
 * `detectChartPattern` returns the single highest-confidence match (or null).
 * This is a descriptive overlay — it does NOT feed the composite score.
 */
import type { Bar, Direction, ChartPattern } from "./types";

const clamp = (x: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, x));
const closes = (b: Bar[]) => b.map((x) => x.close);
const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);

/** Least-squares fit y = a + b·x. Returns slope b, intercept a, and r² (0–1). */
function linfit(xs: number[], ys: number[]): { a: number; b: number; r2: number } {
  const n = xs.length;
  if (n < 2) return { a: ys[0] ?? 0, b: 0, r2: 0 };
  let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
    syy += ys[i] * ys[i];
  }
  const denom = n * sxx - sx * sx;
  const b = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
  const a = (sy - b * sx) / n;
  const ssTot = syy - (sy * sy) / n;
  let ssRes = 0;
  for (let i = 0; i < n; i++) ssRes += (ys[i] - (a + b * xs[i])) ** 2;
  const r2 = ssTot <= 1e-12 ? 0 : clamp(1 - ssRes / ssTot, 0, 1);
  return { a, b, r2 };
}

/** Swing-pivot indices: a high (low) that is the local max (min) over ±k bars. */
function swings(bars: Bar[], k: number): { highs: number[]; lows: number[] } {
  const highs: number[] = [];
  const lows: number[] = [];
  for (let i = k; i < bars.length - k; i++) {
    let isHigh = true;
    let isLow = true;
    for (let j = i - k; j <= i + k; j++) {
      if (j === i) continue;
      if (bars[j].high >= bars[i].high) isHigh = false;
      if (bars[j].low <= bars[i].low) isLow = false;
    }
    if (isHigh) highs.push(i);
    if (isLow) lows.push(i);
  }
  return { highs, lows };
}

/* ------------------------------- triangles ------------------------------- */

function detectTriangle(win: Bar[]): ChartPattern | null {
  const n = win.length;
  if (n < 20) return null;
  const { highs, lows } = swings(win, 2);
  if (highs.length < 2 || lows.length < 2) return null;

  const hf = linfit(highs, highs.map((i) => win[i].high));
  const lf = linfit(lows, lows.map((i) => win[i].low));
  const price = mean(closes(win));
  if (price <= 0) return null;

  const hs = (hf.b / price) * 100; // upper-line slope, %/bar
  const ls = (lf.b / price) * 100; // lower-line slope, %/bar
  const startGap = hf.a - lf.a;
  const endGap = hf.a + hf.b * (n - 1) - (lf.a + lf.b * (n - 1));
  if (endGap <= 0 || startGap <= 0) return null; // lines already crossed
  const converging = endGap < startGap * 0.78;
  if (!converging) return null;

  const FLAT = 0.05; // %/bar treated as horizontal
  const SLOPE = 0.06; // %/bar treated as clearly sloped
  const touches = clamp((highs.length + lows.length - 4) * 8 + 38, 0, 66);
  const fit = ((hf.r2 + lf.r2) / 2) * 28;
  const base = clamp(38 + touches * 0.45 + fit);

  if (Math.abs(hs) < FLAT && ls > SLOPE) {
    return {
      key: "ascending_triangle",
      name: "Ascending Triangle",
      direction: "bullish",
      confidence: clamp(base + 8),
      description: "Flat resistance with rising lows — bullish continuation.",
    };
  }
  if (Math.abs(ls) < FLAT && hs < -SLOPE) {
    return {
      key: "descending_triangle",
      name: "Descending Triangle",
      direction: "bearish",
      confidence: clamp(base + 8),
      description: "Flat support with falling highs — bearish continuation.",
    };
  }
  if (hs < -FLAT && ls > FLAT) {
    const pre = win.slice(0, Math.max(3, Math.floor(n * 0.25)));
    const dir: Direction = mean(closes(pre)) < price ? "bullish" : "bearish";
    return {
      key: "symmetrical_triangle",
      name: "Symmetrical Triangle",
      direction: dir,
      confidence: clamp(base),
      description: "Converging highs and lows — breakout pending; bias follows the prior trend.",
    };
  }
  return null;
}

/* --------------------------------- flags --------------------------------- */

function detectFlag(bars: Bar[]): ChartPattern | null {
  const n = bars.length;
  if (n < 15) return null;
  let best: ChartPattern | null = null;

  for (let F = 4; F <= 12; F++) {
    if (n < F + 6) break;
    const flag = bars.slice(n - F);
    const poleEnd = n - F;
    const fc = linfit(flag.map((_, i) => i), closes(flag));
    const flagMean = mean(closes(flag));
    if (flagMean <= 0) continue;
    const flagSlope = (fc.b / flagMean) * 100;
    const flagHigh = Math.max(...flag.map((b) => b.high));
    const flagLow = Math.min(...flag.map((b) => b.low));
    const flagRange = flagHigh - flagLow;
    const flagAtr = mean(flag.map((b) => b.high - b.low));

    for (let P = 6; P <= 20; P++) {
      const poleStart = poleEnd - P;
      if (poleStart < 0) break;
      const pole = bars.slice(poleStart, poleEnd);
      const a = pole[0].close;
      const z = pole[pole.length - 1].close;
      if (a <= 0) continue;
      const poleRet = (z - a) / a;
      const poleMove = Math.abs(z - a);
      if (poleMove <= 0) continue;
      const poleAtr = mean(pole.map((b) => b.high - b.low));
      const contracts = flagAtr < poleAtr * 0.95;
      if (!contracts || flagRange >= poleMove * 0.6) continue;

      // Bull flag: strong up pole, flag drifts flat/down, shallow retrace.
      if (poleRet >= 0.08 && flagSlope <= 0.05) {
        const retrace = (z - flagLow) / poleMove;
        if (retrace < 0.5) {
          const conf = clamp(46 + poleRet * 110 + (0.5 - retrace) * 36);
          if (!best || conf > best.confidence)
            best = {
              key: "bull_flag",
              name: "Bull Flag",
              direction: "bullish",
              confidence: conf,
              description: "Sharp advance then a tight downward drift — bullish continuation.",
            };
        }
      }
      // Bear flag: mirror image.
      if (poleRet <= -0.08 && flagSlope >= -0.05) {
        const retrace = (flagHigh - z) / poleMove;
        if (retrace < 0.5) {
          const conf = clamp(46 + Math.abs(poleRet) * 110 + (0.5 - retrace) * 36);
          if (!best || conf > best.confidence)
            best = {
              key: "bear_flag",
              name: "Bear Flag",
              direction: "bearish",
              confidence: conf,
              description: "Sharp decline then a tight upward drift — bearish continuation.",
            };
        }
      }
    }
  }
  return best;
}

/* ----------------------------- cup & handle ------------------------------ */

function detectCupHandle(bars: Bar[]): ChartPattern | null {
  const n = bars.length;
  if (n < 40) return null;
  const win = bars.slice(Math.max(0, n - 130));
  const c = closes(win);
  const m = win.length;

  const leftEnd = Math.floor(m * 0.4);
  let leftIdx = 0;
  for (let i = 1; i < leftEnd; i++) if (c[i] > c[leftIdx]) leftIdx = i;

  const handleRoom = Math.max(3, Math.floor(m * 0.08));
  let botIdx = leftIdx;
  for (let i = leftIdx; i < m - handleRoom; i++) if (c[i] < c[botIdx]) botIdx = i;
  if (botIdx <= leftIdx + 3) return null;

  let rightIdx = botIdx;
  for (let i = botIdx; i < m; i++) if (c[i] > c[rightIdx]) rightIdx = i;
  if (rightIdx <= botIdx + 3 || rightIdx >= m - 1) return null;

  const leftPx = c[leftIdx];
  const botPx = c[botIdx];
  const rightPx = c[rightIdx];
  const depth = (leftPx - botPx) / leftPx;
  if (depth < 0.12 || depth > 0.5) return null;

  const rimDiff = Math.abs(rightPx - leftPx) / leftPx;
  if (rimDiff > 0.08) return null;

  const cupLen = rightIdx - leftIdx;
  if (botIdx - leftIdx < cupLen * 0.25 || rightIdx - botIdx < cupLen * 0.25) return null;

  // Rounded (U, not V): several bars hug the bottom rather than one sharp spike.
  const nearBottom = c.slice(leftIdx, rightIdx).filter((v) => v <= botPx * 1.06).length;
  if (nearBottom < Math.max(3, Math.floor(cupLen * 0.15))) return null;

  const base = clamp(40 + (1 - rimDiff / 0.08) * 18 + Math.min(nearBottom, 10));

  // Handle: shallow pullback off the right rim, staying in the cup's upper half.
  const handle = c.slice(rightIdx);
  const handleLow = Math.min(...handle);
  const handleDepth = (rightPx - handleLow) / rightPx;
  const handleOk =
    handle.length >= 2 &&
    handleDepth > 0 &&
    handleDepth < Math.min(0.15, depth * 0.5) &&
    handleLow > botPx + (leftPx - botPx) * 0.5;

  if (handleOk) {
    return {
      key: "cup_handle",
      name: "Cup & Handle",
      direction: "bullish",
      confidence: clamp(base + 14),
      description: "Rounded base recovering to the prior high with a shallow handle — bullish.",
    };
  }
  if (base >= 58) {
    return {
      key: "rounded_bottom",
      name: "Rounded Bottom (Cup)",
      direction: "bullish",
      confidence: clamp(base - 6),
      description: "U-shaped base back near the prior high — handle not yet formed.",
    };
  }
  return null;
}

/* --------------------------------- public -------------------------------- */

/** Best classical chart pattern on the recent price action, or null. */
export function detectChartPattern(bars: Bar[]): ChartPattern | null {
  if (!bars || bars.length < 20) return null;
  const recent = bars.slice(-160);
  const candidates = [
    detectFlag(recent.slice(-30)),
    detectTriangle(recent.slice(-Math.min(50, recent.length))),
    detectCupHandle(recent),
  ].filter((c): c is ChartPattern => !!c && c.confidence >= 45);

  if (!candidates.length) return null;
  candidates.sort((a, b) => b.confidence - a.confidence);
  return candidates[0];
}
