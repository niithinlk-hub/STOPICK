import { NextResponse } from "next/server";
import { z } from "zod";
import { runScan } from "@/lib/engine/scan";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 120;

const schema = z.object({
  country: z.enum(["NSE", "US", "BOTH"]).default("US"),
  source: z.enum(["tier_1", "tier_2", "tier_3", "sample", "manual"]).default("sample"),
  timeframe: z.enum(["1d", "4h", "1h", "15m"]).default("1d"),
  setupMode: z.enum(["breakout", "pullback", "both"]).default("both"),
  minScore: z.number().min(0).max(100).default(75),
  manualSymbols: z.string().optional(),
  limit: z.number().int().positive().max(500).optional(),
  live: z.boolean().optional(),
  includeWatch: z.boolean().optional(),
});

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid scan parameters.", details: parsed.error.flatten() }, { status: 400 });
  }
  try {
    const result = await runScan(parsed.data);
    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Scan failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
