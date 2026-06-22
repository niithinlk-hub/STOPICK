import "server-only";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { getTelegramSettings } from "@/lib/server/telegramSettings";
import { sendTelegram } from "@/lib/telegram";
import { tradingViewUrl } from "@/lib/tradingview";
import type { WatchRow } from "@/lib/engine/types";

const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const DEFAULT_SETS = ["NSE:tier_1", "NSE:tier_2", "NSE:tier_3", "US:tier_1", "US:tier_2", "US:tier_3"];

export interface CoilingAlertResult {
  sent: boolean;
  alerts: number;
  error?: string;
  note?: string;
}

/**
 * Alert when a coiling name's readiness crosses the threshold (default 90) — i.e. it's
 * primed to break. Deduped via `stocker_alerts` so each name pings at most once per day.
 * Pass `rows` (e.g. from the digest scan) to avoid re-scanning; otherwise it fans out to
 * the scan-set workers itself (used by the external intraday trigger).
 */
export async function sendCoilingAlert(opts: {
  origin: string;
  force?: boolean;
  rows?: WatchRow[];
}): Promise<CoilingAlertResult> {
  const cfg = await getTelegramSettings();
  if (!opts.force && !cfg.enabled) return { sent: false, alerts: 0, note: "Telegram disabled." };
  if (!cfg.userId) return { sent: false, alerts: 0, error: "Admin user not found." };
  const min = cfg.coilingMin ?? 90;

  let watch = opts.rows;
  if (!watch) {
    const sets = cfg.sets.length ? cfg.sets : DEFAULT_SETS;
    const secret = process.env.CRON_SECRET;
    const headers = secret ? { authorization: `Bearer ${secret}` } : undefined;
    const results = await Promise.all(
      sets.map(async (set) => {
        const [market, source] = set.split(":");
        // minScore=101 → no setup rows built (we only want watch); live=1 for fresh readiness.
        const url = `${opts.origin}/api/cron/scan-set?market=${market}&source=${source}&minScore=101&limit=600&watch=1&live=1`;
        try {
          const r = await fetch(url, { headers, cache: "no-store" });
          if (!r.ok) return [] as WatchRow[];
          const j = (await r.json()) as { watch?: WatchRow[] };
          return j.watch ?? [];
        } catch {
          return [] as WatchRow[];
        }
      }),
    );
    watch = results.flat();
  }

  const primed = (watch ?? []).filter((w) => w.readiness >= min).sort((a, b) => b.readiness - a.readiness);
  if (!primed.length) return { sent: false, alerts: 0, note: `No coiling name ≥ ${min}.` };

  // Dedup: each ticker pings once per day.
  const sb = getSupabaseAdmin();
  const today = new Date().toISOString().slice(0, 10);
  const keyOf = (w: WatchRow) => `${w.ticker}|${w.market}|coil|${today}`;
  const { data: inserted } = await sb
    .from("stocker_alerts")
    .upsert(
      primed.map((w) => ({ user_id: cfg.userId, alert_key: keyOf(w) })),
      { onConflict: "user_id,alert_key", ignoreDuplicates: true },
    )
    .select("alert_key");
  const freshKeys = new Set((inserted ?? []).map((r: { alert_key: string }) => r.alert_key));
  const fresh = primed.filter((w) => freshKeys.has(keyOf(w)));
  if (!fresh.length) return { sent: false, alerts: 0, note: "Already alerted today." };

  const body = fresh
    .map(
      (w) =>
        `${w.market === "NSE" ? "🇮🇳" : "🇺🇸"} <b>${esc(w.ticker)}</b> ⚡${Math.round(w.readiness)} — ` +
        `trigger ${fmt(w.trigger)} (${w.distancePct.toFixed(1)}% away · tight ${Math.round(w.tightness)})\n  ` +
        `<a href="${tradingViewUrl(w.ticker, w.market)}">TradingView</a>`,
    )
    .join("\n\n");

  const sent = await sendTelegram(
    `🔥 <b>Coiling primed</b> — readiness ≥ ${min}\n<i>Set a buy-stop just above the trigger.</i>\n\n${body}`,
    cfg.chatId ?? undefined,
  );
  return { sent: sent.ok, error: sent.error, alerts: fresh.length };
}
