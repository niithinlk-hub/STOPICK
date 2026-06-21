import "server-only";
import { sendTelegram } from "@/lib/telegram";
import { tradingViewUrl } from "@/lib/tradingview";
import { getTelegramSettings } from "@/lib/server/telegramSettings";
import { getDhanTokenStatus } from "@/lib/server/dhanToken";
import type { Market, ScanRow } from "@/lib/engine/types";

const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Sets scanned for the daily digest: NSE-1/2/3 + US-1/2. Override with TELEGRAM_SETS
// ("NSE:tier_1,NSE:tier_2,..."). US tier_3 is intentionally absent (empty universe).
const DEFAULT_SETS = "NSE:tier_1,NSE:tier_2,NSE:tier_3,US:tier_1,US:tier_2,US:tier_3";
const SET_LABEL: Record<string, string> = {
  "NSE:tier_1": "NSE-1",
  "NSE:tier_2": "NSE-2",
  "NSE:tier_3": "NSE-3",
  "US:tier_1": "US-1",
  "US:tier_2": "US-2",
  "US:tier_3": "US-3",
};

export interface DigestResult {
  sent: boolean;
  error?: string;
  sets: string[];
  scanned: number;
  qualified: number;
  perSet: Record<string, number>;
}

/**
 * Daily setup digest across multiple (market, tier) sets. Fans out to the
 * `/api/cron/scan-set` worker once per set (parallel, each its own 60s invocation),
 * merges + ranks the qualifying rows, and pushes one Telegram message. `provisional`
 * marks the pre-close run (daily candle still forming). `origin` is the deployment
 * base URL used to reach the workers.
 */
export async function sendSetupDigest(opts: {
  provisional?: boolean;
  origin: string;
  force?: boolean;
}): Promise<DigestResult> {
  const { provisional = false, origin, force = false } = opts;
  const cfg = await getTelegramSettings();
  const off = (error: string): DigestResult => ({ sent: false, error, sets: [], scanned: 0, qualified: 0, perSet: {} });
  if (!force && !cfg.enabled) return off("Telegram digest disabled in settings.");
  if (!force && provisional && !cfg.preclose) return off("Pre-close run disabled in settings.");

  const minScore = cfg.minScore;
  const setLimit = Number(process.env.TELEGRAM_SET_LIMIT ?? 600);
  const topN = cfg.topN;
  const sets = cfg.sets.length ? cfg.sets : DEFAULT_SETS.split(",");
  const secret = process.env.CRON_SECRET;
  const headers = secret ? { authorization: `Bearer ${secret}` } : undefined;

  const results = await Promise.all(
    sets.map(async (set) => {
      const [market, source] = set.split(":");
      const url =
        `${origin}/api/cron/scan-set?market=${market}&source=${source}&minScore=${minScore}&limit=${setLimit}` +
        (provisional ? "&live=1" : "");
      try {
        const r = await fetch(url, { headers, cache: "no-store" });
        if (!r.ok) return { set, rows: [] as ScanRow[], scanned: 0 };
        const j = (await r.json()) as { rows?: ScanRow[]; scanned?: number };
        return { set, rows: j.rows ?? [], scanned: j.scanned ?? 0 };
      } catch {
        return { set, rows: [] as ScanRow[], scanned: 0 };
      }
    }),
  );

  const perSet: Record<string, number> = {};
  let scannedTotal = 0;
  const allRows: ScanRow[] = [];
  for (const r of results) {
    perSet[SET_LABEL[r.set] ?? r.set] = r.rows.length;
    scannedTotal += r.scanned;
    allRows.push(...r.rows);
  }
  allRows.sort((a, b) => b.score - a.score);
  const top = allRows.slice(0, topN);

  const tag = provisional ? "pre-close (provisional)" : "EOD";
  const setSummary = Object.entries(perSet)
    .map(([k, v]) => `${k}:${v}`)
    .join(" · ");
  const header =
    `📈 <b>Stocker ${tag}</b> — score ≥ ${minScore}\n` +
    `${allRows.length} setup(s) across ${scannedTotal} scanned · ${setSummary}`;
  const warn = provisional ? "\n<i>⚠️ Daily candle not closed yet — provisional.</i>" : "";

  const mkt = (m: Market) => (m === "NSE" ? "🇮🇳" : "🇺🇸");
  const body = top.length
    ? top
        .map((r) => {
          const cp = r.chartPattern ? ` · ${esc(r.chartPattern)}` : "";
          const plan = [
            r.entry != null ? `entry ${fmt(r.entry)}` : "",
            r.stop != null ? `SL ${fmt(r.stop)}` : "",
            r.target2r != null ? `T ${fmt(r.target2r)}` : "",
            r.rrRatio != null ? `${r.rrRatio.toFixed(1)}R` : "",
          ]
            .filter(Boolean)
            .join(" · ");
          const url = tradingViewUrl(r.ticker, r.market);
          return (
            `${mkt(r.market)} <b>${esc(r.ticker)}</b> ${r.grade} ${Math.round(r.score)} ` +
            `(${esc(r.setupFamily)}${cp})\n  ${plan}\n  <a href="${url}">TradingView</a>`
          );
        })
        .join("\n\n")
    : "No setups cleared the score.";

  // Dhan token health reminder (only when NSE sets are in play).
  let dhanWarn = "";
  if (sets.some((s) => s.startsWith("NSE:"))) {
    const st = await getDhanTokenStatus();
    if (st.expired) {
      dhanWarn = "\n⚠️ <b>Dhan token expired</b> — NSE is on Yahoo fallback. Send <code>/access &lt;token&gt;</code> or update it in Settings.";
    } else if (st.hoursLeft != null && st.hoursLeft <= 12) {
      dhanWarn = `\n⏳ Dhan token expires in ~${st.hoursLeft}h — refresh soon via <code>/access &lt;token&gt;</code>.`;
    }
  }

  const more = allRows.length > top.length ? `\n\n…and ${allRows.length - top.length} more (top ${topN} shown).` : "";
  const sent = await sendTelegram(`${header}${warn}${dhanWarn}\n\n${body}${more}`, cfg.chatId ?? undefined);
  return { sent: sent.ok, error: sent.error, sets, scanned: scannedTotal, qualified: allRows.length, perSet };
}
