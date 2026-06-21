import { NextResponse } from "next/server";
import { z } from "zod";
import { fetchOhlcv } from "@/lib/data/yahoo";
import { mapWithConcurrency } from "@/lib/data/cache";
import { resolveRecords } from "@/lib/data/universe";
import { analyzeConfluence, maxStopPctFor } from "@/lib/engine/confluence";
import { CONFIG } from "@/lib/config";
import type { ConfluenceRow } from "@/lib/engine/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 120;

const schema = z.object({
  country: z.enum(["NSE", "US", "BOTH"]).default("US"),
  source: z.enum(["tier_1", "tier_2", "tier_3", "sample", "manual"]).default("sample"),
  timeframe: z.enum(["1d", "4h", "1h", "15m"]).default("1d"),
  side: z.enum(["long", "short", "both"]).default("long"),
  manualSymbols: z.string().optional(),
  limit: z.number().int().positive().max(500).optional(),
});

const MODE_LABEL: Record<string, string> = {
  "1d": "BTST / swing (daily)",
  "4h": "Intraday (4H)",
  "1h": "Intraday (1H)",
  "15m": "Intraday (15m)",
};

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid parameters.", details: parsed.error.flatten() }, { status: 400 });
  }
  const { country, source, timeframe, side, manualSymbols, limit } = parsed.data;

  try {
    const startedAt = Date.now();
    const records = resolveRecords({
      country,
      source,
      manualSymbols,
      maxSymbols: Math.min(limit ?? CONFIG.runtime.maxSymbolsPerScan, CONFIG.runtime.maxSymbolsPerScan),
    });

    const failures: Record<string, string> = {};
    const rows: ConfluenceRow[] = [];
    const maxStopPct = maxStopPctFor(timeframe);

    await mapWithConcurrency(records, 8, async (record) => {
      try {
        const bars = await fetchOhlcv(record.symbol, timeframe, 300);
        if (!bars.length) {
          failures[record.symbol] = "No data.";
          return;
        }
        const row = analyzeConfluence(bars, {
          ticker: record.display,
          market: record.market,
          timeframe,
          maxStopPct,
        });
        if (row) rows.push(row);
      } catch (err) {
        failures[record.symbol] = err instanceof Error ? err.message : String(err);
      }
    });

    // Keep only firing signals matching the requested side; rank by confidence.
    const wanted = rows.filter((r) => r.signal && (side === "both" || r.side === side));
    wanted.sort((a, b) => b.confidence - a.confidence);

    const successful = records.length - Object.keys(failures).length;
    return NextResponse.json({
      rows: wanted,
      scannedSymbols: records.length,
      successfulSymbols: successful,
      signalCount: wanted.length,
      failures,
      notes: [
        `Strategy: Stochastic(14,3,3) turn + RSI>50/<50 filter + MACD line/signal cross. Stop at swing ${timeframe === "1d" ? "low/high" : "level"}, 1.5R target.`,
        `Max-stop cap for this timeframe: ${(maxStopPct * 100).toFixed(1)}% (wider stops are penalized, not hidden).`,
      ],
      timeframe,
      mode: MODE_LABEL[timeframe] ?? timeframe,
      generatedAt: startedAt,
      elapsedMs: Date.now() - startedAt,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Confluence scan failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
