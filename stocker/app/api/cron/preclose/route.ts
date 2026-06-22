import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { sendSetupDigest } from "@/lib/server/digest";
import { sendCoilingAlert } from "@/lib/server/coilingAlert";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

/**
 * Pre-close heads-up — Vercel cron fires ~30 min before NSE close (15:00 IST =
 * 09:30 UTC). The daily candle is still forming, so the message is flagged
 * provisional; it lets you act in the last 30 minutes of the same session. Also fires
 * the ≥threshold "coiling primed" alert, reusing this run's watch rows (no extra scan).
 */
export async function GET(req: Request) {
  if (!cronAuthorized(req)) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const origin = new URL(req.url).origin;
  const result = await sendSetupDigest({ provisional: true, origin });
  const coiling = await sendCoilingAlert({ origin, rows: result.watch });
  return NextResponse.json({ ...result, coiling });
}
