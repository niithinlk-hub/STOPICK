import { NextResponse } from "next/server";
import { createSupabaseServer } from "@/lib/supabase/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { isAdminEmail } from "@/lib/server/admins";
import { resolveRecords } from "@/lib/data/universe";
import { fetchOhlcv } from "@/lib/data/yahoo";
import { ensureScripMap, dhanEnabled } from "@/lib/data/dhan";
import { mapWithConcurrency } from "@/lib/data/cache";
import { validateEngine } from "@/lib/engine/validate";
import { CONFIG } from "@/lib/config";
import type { Bar, Market, ScoringProfile, UniverseSource } from "@/lib/engine/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

async function authorized(req: Request): Promise<boolean> {
  if (cronAuthorized(req)) return true;
  try {
    const {
      data: { user },
    } = await createSupabaseServer().auth.getUser();
    return Boolean(user && isAdminEmail(user.email));
  } catch {
    return false;
  }
}

/**
 * Validate the screening engine: does grade actually predict forward return?
 * Walks daily history with no lookahead over a sample of the chosen universe and reports
 * per-grade win-rate / expectancy. `liqWeight` (optional) re-runs with an experimental
 * liquidity weight so calibration can be A/B-tested without touching the live scanner.
 */
export async function GET(req: Request) {
  if (!(await authorized(req))) return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const u = new URL(req.url);
  const market = (u.searchParams.get("market") === "US" ? "US" : "NSE") as Market;
  const source = (u.searchParams.get("source") || "tier_1") as UniverseSource;
  const sample = Math.min(Number(u.searchParams.get("sample") ?? 20), 40);
  // Default 20 bars (~4 trading weeks). A 10-bar horizon almost never lets a 2R target fill
  // (intrabar hit rate ~1%), which made the strategy look worse than its real swing horizon.
  const horizon = Math.min(Number(u.searchParams.get("horizon") ?? 20), 40);
  const liqWeight = u.searchParams.get("liqWeight");
  const targetRParam = u.searchParams.get("targetR");
  const stopAtrParam = u.searchParams.get("stopAtr");
  const targetR = targetRParam != null && Number.isFinite(Number(targetRParam)) ? Number(targetRParam) : undefined;
  const stopAtrMult = stopAtrParam != null && Number.isFinite(Number(stopAtrParam)) ? Number(stopAtrParam) : undefined;

  if (dhanEnabled()) await ensureScripMap();
  const records = resolveRecords({ country: market, source, maxSymbols: sample }).slice(0, sample);
  const benchSym = CONFIG.benchmarkMap[market]?.broad ?? (market === "NSE" ? "^NSEI" : "SPY");

  const [bench, items] = await Promise.all([
    fetchOhlcv(benchSym, "1d", 600),
    (async () => {
      const out: { symbol: string; bars: Bar[] }[] = [];
      await mapWithConcurrency(records, 8, async (r) => {
        const bars = await fetchOhlcv(r.symbol, "1d", 600);
        if (bars.length >= 300) out.push({ symbol: r.display, bars });
      });
      return out;
    })(),
  ]);

  // Optional experimental profile — isolated to this tool, never touches the live scanner.
  // `weights=key:val,key:val` overrides component weights; `thresholds=A+:88,A:80,B:70,C:60`
  // overrides grade cutoffs; `liqWeight` kept for back-compat. Lets weights/thresholds be
  // A/B-tested by query string without a redeploy.
  const base = CONFIG.scoringProfiles.bullish_breakout;
  const weights: Record<string, number> = { ...base.weights };
  const gradeThresholds: Record<string, number> = { ...base.gradeThresholds };
  if (liqWeight != null && Number.isFinite(Number(liqWeight))) weights.liquidity = Number(liqWeight);
  const wParam = u.searchParams.get("weights");
  if (wParam) {
    for (const pair of wParam.split(",")) {
      const idx = pair.indexOf(":");
      if (idx <= 0) continue;
      const k = pair.slice(0, idx).trim();
      const v = Number(pair.slice(idx + 1));
      if (k in weights && Number.isFinite(v)) weights[k] = v;
    }
  }
  const tParam = u.searchParams.get("thresholds");
  if (tParam) {
    for (const pair of tParam.split(",")) {
      const idx = pair.lastIndexOf(":");
      if (idx <= 0) continue;
      const k = pair.slice(0, idx).trim();
      const v = Number(pair.slice(idx + 1));
      if (Number.isFinite(v)) gradeThresholds[k] = v;
    }
  }
  const overridden = wParam != null || tParam != null || (liqWeight != null && Number.isFinite(Number(liqWeight)));
  const profileOverride: ScoringProfile | undefined = overridden
    ? ({ ...base, weights, gradeThresholds } as unknown as ScoringProfile)
    : undefined;

  const result = validateEngine(items, bench, market, { horizonBars: horizon, profileOverride, targetR, stopAtrMult });
  return NextResponse.json({
    market,
    source,
    sampleRequested: sample,
    sampleWithData: items.length,
    horizon,
    targetR: targetR ?? 2,
    stopAtr: stopAtrMult ?? null,
    liqWeight: liqWeight != null ? Number(liqWeight) : CONFIG.scoringProfiles.bullish_breakout.weights.liquidity,
    ...result,
  });
}
