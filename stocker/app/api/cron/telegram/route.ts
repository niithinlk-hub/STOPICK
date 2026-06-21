import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { sendSetupDigest } from "@/lib/server/digest";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

/**
 * Confirmed EOD digest — Vercel cron fires this just after NSE close (16:00 IST =
 * 10:30 UTC). The daily candle is final, so these are actionable for AMO / next open.
 */
export async function GET(req: Request) {
  if (!cronAuthorized(req)) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const result = await sendSetupDigest({ provisional: false, origin: new URL(req.url).origin });
  return NextResponse.json(result);
}
