import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { sendCoilingAlert } from "@/lib/server/coilingAlert";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

/**
 * Pre-breakout "coiling primed" alert (readiness ≥ threshold, default 90). Self-scans and
 * dedups once/day. Built for an EXTERNAL scheduler (cron-job.org every ~20 min during NSE/US
 * session) so you get pinged the moment a name crosses the threshold — Vercel Hobby crons
 * can't run intraday. Also auto-fired once at the pre-close run.
 */
export async function GET(req: Request) {
  if (!cronAuthorized(req)) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const result = await sendCoilingAlert({ origin: new URL(req.url).origin, force: true });
  return NextResponse.json(result);
}
