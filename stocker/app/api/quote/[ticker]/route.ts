import { NextResponse } from "next/server";
import { fetchOhlcv } from "@/lib/data/yahoo";
import { normalizeSymbol, displaySymbol } from "@/lib/data/universe";
import type { Market } from "@/lib/engine/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Lightweight last-price lookup for marking watchlist / paper-trade positions. */
export async function GET(request: Request, { params }: { params: { ticker: string } }) {
  const url = new URL(request.url);
  const market = (url.searchParams.get("market") === "NSE" ? "NSE" : "US") as Market;
  const symbol = normalizeSymbol(decodeURIComponent(params.ticker), market);

  try {
    const bars = await fetchOhlcv(symbol, "1d", 5);
    if (!bars.length) {
      return NextResponse.json({ error: "No data for symbol." }, { status: 404 });
    }
    const last = bars[bars.length - 1];
    const prev = bars.length > 1 ? bars[bars.length - 2] : null;
    return NextResponse.json({
      ticker: displaySymbol(symbol),
      market,
      price: last.close,
      prevClose: prev ? prev.close : last.open,
      asOf: last.time,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Quote failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
