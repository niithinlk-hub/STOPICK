import type { Market } from "@/lib/engine/types";

/**
 * Deep link into Dhan's TradingView trader (tv.dhan.co) for a ticker. Dhan trades NSE/BSE
 * only, so US names return null (no Trade-on-Dhan button). The symbol param mirrors the
 * TradingView format; if Dhan ignores it the link still lands the user in their trader.
 */
export function dhanTradeUrl(ticker: string, market: Market): string | null {
  if (market !== "NSE") return null;
  const t = ticker.trim().toUpperCase().replace(/\.NS$/, "");
  return `https://tv.dhan.co/?symbol=${encodeURIComponent(`NSE:${t}`)}`;
}
