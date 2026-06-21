import { NextResponse } from "next/server";
import { sendSetupDigest } from "@/lib/server/digest";
import { sendTelegram } from "@/lib/telegram";
import { getTelegramSettings } from "@/lib/server/telegramSettings";
import { storeDhanToken } from "@/lib/server/dhanToken";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const HELP =
  "<b>Stocker bot</b>\n" +
  "/run — scan all your selected sets now and send the digest\n" +
  "/preclose — provisional pre-close scan (candle still forming)\n" +
  "/access <token> — update the Dhan NSE data token (expires ~daily)\n" +
  "/help — this message\n\n" +
  "Auto: pre-close ~3 PM IST, confirmed EOD ~4 PM IST (Mon–Fri). Configure sets, score and toggles in the app → Settings → Telegram.";

/**
 * Telegram webhook. Telegram POSTs updates here; we act on the owner's commands.
 * Verified by the secret token Telegram echoes in `X-Telegram-Bot-Api-Secret-Token`
 * and restricted to the configured chat. Always returns 200 so Telegram doesn't retry.
 */
export async function POST(req: Request) {
  const secret = process.env.TELEGRAM_WEBHOOK_SECRET;
  if (secret && req.headers.get("x-telegram-bot-api-secret-token") !== secret) {
    return NextResponse.json({ ok: true });
  }

  let update: unknown;
  try {
    update = await req.json();
  } catch {
    return NextResponse.json({ ok: true });
  }

  const msg = (update as { message?: unknown; edited_message?: unknown })?.message ??
    (update as { edited_message?: unknown })?.edited_message;
  const m = msg as { text?: string; chat?: { id?: number | string } } | undefined;
  const text = (m?.text ?? "").trim();
  const chatId = m?.chat?.id != null ? String(m.chat.id) : null;
  if (!chatId || !text) return NextResponse.json({ ok: true });

  const cfg = await getTelegramSettings();
  // Only respond to the configured owner chat.
  if (cfg.chatId && chatId !== String(cfg.chatId)) return NextResponse.json({ ok: true });

  const cmd = text.split(/\s+/)[0].toLowerCase().replace(/@.*$/, "");
  const origin = new URL(req.url).origin;

  try {
    if (cmd === "/run" || cmd === "/scan") {
      await sendTelegram("⏳ Scanning all selected sets… (~1 min)", chatId);
      await sendSetupDigest({ provisional: false, origin, force: true });
    } else if (cmd === "/preclose") {
      await sendTelegram("⏳ Running pre-close (provisional) scan…", chatId);
      await sendSetupDigest({ provisional: true, origin, force: true });
    } else if (cmd === "/access") {
      const parts = text.split(/\s+/);
      if (parts.length < 2) {
        await sendTelegram("Usage: <code>/access &lt;your Dhan token&gt;</code>", chatId);
      } else {
        const r = await storeDhanToken(parts.slice(1).join(""));
        await sendTelegram(
          r.ok ? `✅ Dhan token saved — expires ${new Date((r.exp ?? 0) * 1000).toUTCString()}.` : `⚠️ ${r.error}`,
          chatId,
        );
      }
    } else if (cmd === "/help" || cmd === "/start") {
      await sendTelegram(HELP, chatId);
    }
  } catch (err) {
    await sendTelegram(`⚠️ Command failed: ${err instanceof Error ? err.message : "error"}`, chatId);
  }

  return NextResponse.json({ ok: true });
}
