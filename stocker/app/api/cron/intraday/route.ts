import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { getTelegramSettings } from "@/lib/server/telegramSettings";
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

/**
 * Intraday watchlist alerts. Scans the saved watchlist on the configured intraday
 * timeframe with the RSI-Confluence engine and pings Telegram on a fresh long trigger.
 * Deduped via `stocker_alerts` (once per symbol/timeframe/day). NOT a Vercel cron —
 * drive it from an external scheduler every ~20 min during NSE/US session hours.
 */
export async function GET(req: Request) {
  if (!cronAuthorized(req)) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const cfg = await getTelegramSettings();
  if (!cfg.intradayEnabled) return NextResponse.json({ sent: false, note: "Intraday alerts disabled in settings." });
  if (!cfg.userId) return NextResponse.json({ sent: false, error: "Watchlist user not found." }, { status: 404 });

  const tf = (cfg.intradayTf as Timeframe) || "15m";
  const sb = getSupabaseAdmin();
  const { data: wl } = await sb.from("stocker_watchlist").select("symbol, market").eq("user_id", cfg.userId);
  const items = (wl ?? []) as { symbol: string; market: Market }[];
  if (!items.length) return NextResponse.json({ sent: false, alerts: 0, note: "Watchlist is empty." });

  const maxStopPct = maxStopPctFor(tf);
  const fires: ConfluenceRow[] = [];
  await mapWithConcurrency(items, 5, async (it) => {
    try {
      const bars = await fetchOhlcv(normalizeSymbol(it.symbol, it.market), tf, 300);
      if (!bars.length) return;
      const row = analyzeConfluence(bars, { ticker: it.symbol, market: it.market, timeframe: tf, maxStopPct });
      if (row && row.signal && row.side === "long") fires.push(row);
    } catch {
      /* skip */
    }
  });
  if (!fires.length) return NextResponse.json({ sent: false, alerts: 0, scanned: items.length });

  // Dedup: alert each symbol at most once per day per timeframe.
  const today = new Date().toISOString().slice(0, 10);
  const keyOf = (f: ConfluenceRow) => `${f.ticker}|${f.market}|${tf}|${today}`;
  const { data: inserted } = await sb
    .from("stocker_alerts")
    .upsert(
      fires.map((f) => ({ user_id: cfg.userId, alert_key: keyOf(f) })),
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

  const sent = await sendTelegram(`⚡ <b>Intraday watchlist trigger</b> (${tf})\n\n${body}`, cfg.chatId ?? undefined);
  return NextResponse.json({ sent: sent.ok, error: sent.error, alerts: fresh.length, scanned: items.length });
}
