import { NextResponse } from "next/server";
import { analyzeSymbol } from "@/lib/engine/scan";
import { normalizeSymbol } from "@/lib/data/universe";
import type { Market, SetupFamily, SymbolDetailResponse, Timeframe } from "@/lib/engine/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TIMEFRAMES: Timeframe[] = ["1d", "4h", "1h", "15m"];

export async function GET(request: Request, { params }: { params: { ticker: string } }) {
  const url = new URL(request.url);
  const market = (url.searchParams.get("market") === "NSE" ? "NSE" : "US") as Market;
  const tfParam = url.searchParams.get("timeframe") as Timeframe | null;
  const timeframe: Timeframe = tfParam && TIMEFRAMES.includes(tfParam) ? tfParam : "1d";
  const family = (url.searchParams.get("family") === "pullback" ? "pullback" : "breakout") as SetupFamily;
  const symbol = normalizeSymbol(decodeURIComponent(params.ticker), market);

  try {
    const { bars, setup } = await analyzeSymbol(symbol, market, timeframe, family);
    const payload: SymbolDetailResponse = {
      ticker: setup?.ticker ?? params.ticker,
      market,
      timeframe,
      bars,
      setup,
    };
    return NextResponse.json(payload);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Symbol lookup failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
