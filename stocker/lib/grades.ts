import type { Direction, Grade } from "./engine/types";

/**
 * Visual mappings for grades / direction / score. Tailwind classes use semantic
 * tokens (see DESIGN.md). Always pair color with text/icon — never color alone.
 */
export const GRADE_STYLES: Record<Grade, { text: string; bg: string; ring: string; label: string }> = {
  "A+": { text: "text-bull", bg: "bg-bull/15", ring: "ring-bull/40", label: "A+" },
  A: { text: "text-bull", bg: "bg-bull/10", ring: "ring-bull/30", label: "A" },
  B: { text: "text-info", bg: "bg-info/10", ring: "ring-info/30", label: "B" },
  C: { text: "text-warn", bg: "bg-warn/10", ring: "ring-warn/30", label: "C" },
  Reject: { text: "text-faint", bg: "bg-faint/10", ring: "ring-faint/20", label: "Reject" },
};

export const DIRECTION_STYLES: Record<Direction, { text: string; label: string }> = {
  bullish: { text: "text-bull", label: "Bullish" },
  bearish: { text: "text-bear", label: "Bearish" },
  neutral: { text: "text-muted", label: "Neutral" },
};

/** Color a 0–100 score: faint → info → bull. Returns a tailwind text class. */
export function scoreColor(score: number): string {
  if (score >= 85) return "text-bull";
  if (score >= 75) return "text-info";
  if (score >= 65) return "text-warn";
  return "text-faint";
}

/** Returns an `rgb(var(--token))` string for canvas/SVG use (charts, gauges). */
export function scoreRgb(score: number): string {
  if (score >= 85) return "rgb(var(--bull))";
  if (score >= 75) return "rgb(var(--info))";
  if (score >= 65) return "rgb(var(--warn))";
  return "rgb(var(--faint))";
}
