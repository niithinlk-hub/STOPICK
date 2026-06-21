import { NextResponse } from "next/server";
import { fetchOhlcv } from "@/lib/data/yahoo";
import { runBacktest } from "@/lib/engine/backtest";
import { normalizeSymbol } from "@/lib/data/universe";
import { CONFIG } from "@/lib/config";
import type { Market } from "@/lib/engine/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET(request: Request, { params }: { params: { ticker: string } }) {
  const url = new URL(request.url);
  const market = (url.searchParams.get("market") === "NSE" ? "NSE" : "US") as Market;
  const symbol = normalizeSymbol(decodeURIComponent(params.ticker), market);

  try {
    const bars = await fetchOhlcv(symbol, "1d", 600);
    if (!bars.length) {
      return NextResponse.json({ error: "No data available for this symbol." }, { status: 404 });
    }
    const result = runBacktest(bars, {
      slippageBps: CONFIG.runtime.slippageBps,
      brokerageBps: CONFIG.runtime.brokerageBps,
      taxesBps: CONFIG.runtime.taxesBps,
    });
    return NextResponse.json({ ticker: symbol, market, ...result });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backtest failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
