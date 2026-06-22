import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { runScan } from "@/lib/engine/scan";
import type { Country, UniverseSource } from "@/lib/engine/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

/**
 * Worker: scan ONE (market, tier) set and return its qualifying rows. The digest
 * orchestrator fans out to several of these in parallel so each ~500-name set runs
 * in its own 60s serverless invocation (a single 2000-name scan would blow the cap).
 * Internal-only — guarded by the same CRON_SECRET.
 */
export async function GET(req: Request) {
  if (!cronAuthorized(req)) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const u = new URL(req.url);
  const market = (u.searchParams.get("market") === "US" ? "US" : "NSE") as Country;
  const source = (u.searchParams.get("source") || "tier_1") as UniverseSource;
  const minScore = Number(u.searchParams.get("minScore") ?? 75);
  const limit = Number(u.searchParams.get("limit") ?? 600);
  const live = u.searchParams.get("live") === "1";
  const watch = u.searchParams.get("watch") === "1";

  try {
    const r = await runScan({ country: market, source, timeframe: "1d", setupMode: "both", minScore, limit, live, includeWatch: watch });
    return NextResponse.json({
      market,
      source,
      scanned: r.scannedSymbols,
      successful: r.successfulSymbols,
      rows: r.rows.filter((x) => x.score >= minScore),
      watch: r.watch ?? [],
    });
  } catch (err) {
    return NextResponse.json({ market, source, scanned: 0, rows: [], error: String(err) });
  }
}
