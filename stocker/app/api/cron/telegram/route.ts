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
  const origin = new URL(req.url).origin;

  // Weekly NSE scrip-map refresh (Mondays IST). Runs as its own invocation so the CSV
  // parse gets a fresh 60s budget and never eats the digest's. Best-effort.
  try {
    const isMonday = new Date().toLocaleDateString("en-US", { weekday: "short", timeZone: "Asia/Kolkata" }) === "Mon";
    if (isMonday) {
      const secret = process.env.CRON_SECRET;
      await fetch(`${origin}/api/cron/refresh-scrip`, {
        headers: secret ? { authorization: `Bearer ${secret}` } : undefined,
        cache: "no-store",
      });
    }
  } catch {
    /* refresh failure must never block the digest */
  }

  const result = await sendSetupDigest({ provisional: false, origin });
  return NextResponse.json(result);
}
