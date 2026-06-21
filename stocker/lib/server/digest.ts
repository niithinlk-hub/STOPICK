import "server-only";
import { runScan } from "@/lib/engine/scan";
import { sendTelegram } from "@/lib/telegram";
import { tradingViewUrl } from "@/lib/tradingview";
import type { Country, UniverseSource } from "@/lib/engine/types";

const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export interface DigestResult {
  sent: boolean;
  error?: string;
  country: Country;
  source: UniverseSource;
  minScore: number;
  scanned: number;
  qualified: number;
}

/**
 * Run the daily setup scan and push the qualifying names to Telegram. `provisional`
 * marks the pre-close run (daily candle still forming) so the message warns it isn't
 * a confirmed close. Universe/threshold come from env (defaults NSE / tier_1 / 75 / 150).
 */
export async function sendSetupDigest(opts: { provisional?: boolean } = {}): Promise<DigestResult> {
  const country = ((process.env.TELEGRAM_SCAN_COUNTRY as Country) || "NSE") as Country;
  const source = ((process.env.TELEGRAM_SCAN_SOURCE as UniverseSource) || "tier_1") as UniverseSource;
  const minScore = Number(process.env.TELEGRAM_MIN_SCORE ?? 75);
  const limit = Number(process.env.TELEGRAM_SCAN_LIMIT ?? 150);
  const base = { country, source, minScore, scanned: 0, qualified: 0 };

  let result;
  try {
    result = await runScan({ country, source, timeframe: "1d", setupMode: "both", minScore, limit });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Scan failed.";
    await sendTelegram(`⚠️ <b>Stocker</b> scan failed: ${esc(message)}`);
    return { ...base, sent: false, error: message };
  }

  const rows = result.rows.filter((r) => r.score >= minScore).slice(0, 25);
  const tag = opts.provisional ? "pre-close (provisional)" : "EOD";
  const header =
    `📈 <b>Stocker ${tag}</b> — ${country} · score ≥ ${minScore}\n` +
    `${rows.length} setup(s) · ${result.successfulSymbols}/${result.scannedSymbols} scanned`;
  const warn = opts.provisional
    ? "\n<i>⚠️ Daily candle not closed yet — provisional, may change by 3:30 PM.</i>"
    : "";

  const body = rows.length
    ? rows
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
            `<b>${esc(r.ticker)}</b> ${r.grade} ${Math.round(r.score)} ` +
            `(${esc(r.setupFamily)}${cp})\n  ${plan}\n  <a href="${url}">TradingView</a>`
          );
        })
        .join("\n\n")
    : "No setups cleared the score.";

  const sent = await sendTelegram(`${header}${warn}\n\n${body}`);
  return { ...base, sent: sent.ok, error: sent.error, scanned: result.scannedSymbols, qualified: rows.length };
}
