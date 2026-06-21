import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { fetchOhlcv } from "@/lib/data/yahoo";
import { normalizeSymbol } from "@/lib/data/universe";
import { analyzeConfluence, maxStopPctFor } from "@/lib/engine/confluence";
import { mapWithConcurrency } from "@/lib/data/cache";
import { sendTelegram } from "@/lib/telegram";
import { tradingViewUrl } from "@/lib/tradingview";
import type { ConfluenceRow, Market, Timeframe } from "@/lib/engine/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

type Admin = ReturnType<typeof getSupabaseAdmin>;

/** Resolve the user whose watchlist drives the alerts (first matching ADMIN_EMAILS account). */
async function watchlistUserId(sb: Admin): Promise<string | null> {
  const emails = (process.env.ADMIN_EMAILS || "niithin.lk@gmail.com,admin@stocker.app")
    .split(",")
    .map((e) => e.trim().toLowerCase());
  const { data } = await sb.auth.admin.listUsers({ page: 1, perPage: 200 });
  const u = (data?.users ?? []).find((x) => emails.includes((x.email ?? "").toLowerCase()));
  return u?.id ?? null;
}

/**
 * Intraday watchlist alerts. Scans the saved watchlist on an intraday timeframe with
 * the RSI-Confluence engine and pings Telegram the moment a long trigger fires.
 * Deduped via `stocker_alerts` (once per symbol/timeframe/day) so a 15–30 min external
 * cron can hit this all session without spamming. NOT a Vercel cron — Hobby crons only
 * run daily; drive it from cron-job.org / a GitHub Action every ~20 min during 09:15–15:30 IST.
 */
export async function GET(req: Request) {
  if (!cronAuthorized(req)) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const tf = ((process.env.TELEGRAM_INTRADAY_TF as Timeframe) || "15m") as Timeframe;
  const sb = getSupabaseAdmin();

  const userId = await watchlistUserId(sb);
  if (!userId) return NextResponse.json({ sent: false, error: "Watchlist user not found." }, { status: 404 });

  const { data: wl } = await sb.from("stocker_watchlist").select("symbol, market").eq("user_id", userId);
  const items = (wl ?? []) as { symbol: string; market: Market }[];
  if (!items.length) return NextResponse.json({ sent: false, alerts: 0, note: "Watchlist is empty." });

  const maxStopPct = maxStopPctFor(tf);
  const fires: ConfluenceRow[] = [];
  await mapWithConcurrency(items, 5, async (it) => {
    try {
      const sym = normalizeSymbol(it.symbol, it.market);
      const bars = await fetchOhlcv(sym, tf, 300);
      if (!bars.length) return;
      const row = analyzeConfluence(bars, { ticker: it.symbol, market: it.market, timeframe: tf, maxStopPct });
      if (row && row.signal && row.side === "long") fires.push(row);
    } catch {
      /* skip */
    }
  });

  if (!fires.length) return NextResponse.json({ sent: false, alerts: 0, scanned: items.length });

  // Dedup: alert each symbol at most once per day per timeframe. ON CONFLICT DO NOTHING +
  // .select() returns only the rows we actually inserted = the not-yet-alerted triggers.
  const today = new Date().toISOString().slice(0, 10);
  const keyOf = (f: ConfluenceRow) => `${f.ticker}|${f.market}|${tf}|${today}`;
  const { data: inserted } = await sb
    .from("stocker_alerts")
    .upsert(
      fires.map((f) => ({ user_id: userId, alert_key: keyOf(f) })),
      { onConflict: "user_id,alert_key", ignoreDuplicates: true },
    )
    .select("alert_key");
  const freshKeys = new Set((inserted ?? []).map((r: { alert_key: string }) => r.alert_key));
  const fresh = fires.filter((f) => freshKeys.has(keyOf(f)));

  if (!fresh.length) return NextResponse.json({ sent: false, alerts: 0, note: "Already alerted today." });

  const body = fresh
    .map((f) => {
      const plan = [
        f.entry != null ? `entry ${fmt(f.entry)}` : "",
        f.stop != null ? `SL ${fmt(f.stop)}` : "",
        f.target != null ? `T ${fmt(f.target)}` : "",
      ]
        .filter(Boolean)
        .join(" · ");
      return `<b>${esc(f.ticker)}</b> ${tf} confluence ${Math.round(f.confidence)}\n  ${plan}\n  <a href="${tradingViewUrl(f.ticker, f.market)}">TradingView</a>`;
    })
    .join("\n\n");

  const sent = await sendTelegram(`⚡ <b>Intraday watchlist trigger</b> (${tf})\n\n${body}`);
  return NextResponse.json({ sent: sent.ok, error: sent.error, alerts: fresh.length, scanned: items.length });
}
