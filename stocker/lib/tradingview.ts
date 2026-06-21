import type { Market } from "@/lib/engine/types";

// Exchanges TradingView recognises by prefix. US listings resolve fine from the
// bare symbol, but Indian names MUST be prefixed with NSE: or they won't be found.
const TV_US_EXCHANGES = new Set(["NASDAQ", "NYSE", "AMEX", "ARCA", "BATS", "OTC"]);

/** TradingView symbol string, e.g. "NSE:RELIANCE" or "NASDAQ:AAPL" / "AAPL". */
export function tradingViewSymbol(ticker: string, market: Market, exchange?: string): string {
  const t = ticker.trim().toUpperCase().replace(/\.NS$/, "");
  if (market === "NSE") return `NSE:${t}`;
  const ex = (exchange ?? "").toUpperCase();
  return TV_US_EXCHANGES.has(ex) ? `${ex}:${t}` : t;
}

/** Deep link straight to a ticker's TradingView chart (not the TV home page). */
export function tradingViewUrl(ticker: string, market: Market, exchange?: string): string {
  const sym = tradingViewSymbol(ticker, market, exchange);
  return `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(sym)}`;
}
