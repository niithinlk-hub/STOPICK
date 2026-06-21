import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { sendSetupDigest } from "@/lib/server/digest";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

/**
 * Pre-close heads-up — Vercel cron fires ~30 min before NSE close (15:00 IST =
 * 09:30 UTC). The daily candle is still forming, so the message is flagged
 * provisional; it lets you act in the last 30 minutes of the same session.
 */
export async function GET(req: Request) {
  if (!cronAuthorized(req)) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const result = await sendSetupDigest({ provisional: true, origin: new URL(req.url).origin });
  return NextResponse.json(result);
}
