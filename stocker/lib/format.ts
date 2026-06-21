/** Locale-aware formatters for the UI. Numbers use tabular figures in CSS. */

export function fmtNumber(value: number | null | undefined, dp = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

export function fmtPrice(value: number | null | undefined, market: "NSE" | "US" = "US"): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const symbol = market === "NSE" ? "₹" : "$";
  return `${symbol}${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function fmtPct(value: number | null | undefined, dp = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(dp)}%`;
}

export function fmtCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function fmtSignedPct(value: number | null | undefined, dp = 2): string {
  return fmtPct(value, dp);
}
