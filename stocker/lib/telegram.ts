/**
 * Telegram Bot send helper. Reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from the
 * environment (server-only). Splits long messages under Telegram's 4096-char cap.
 */
const API = "https://api.telegram.org";

function chunkText(text: string, max = 3800): string[] {
  if (text.length <= max) return [text];
  const lines = text.split("\n");
  const out: string[] = [];
  let cur = "";
  for (const ln of lines) {
    if ((cur ? cur.length + 1 : 0) + ln.length > max) {
      if (cur) out.push(cur);
      cur = ln;
    } else {
      cur = cur ? `${cur}\n${ln}` : ln;
    }
  }
  if (cur) out.push(cur);
  return out;
}

export async function sendTelegram(text: string): Promise<{ ok: boolean; error?: string }> {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId) return { ok: false, error: "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set." };

  for (const chunk of chunkText(text)) {
    const res = await fetch(`${API}/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        text: chunk,
        parse_mode: "HTML",
        disable_web_page_preview: true,
      }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      return { ok: false, error: `Telegram ${res.status}: ${detail.slice(0, 300)}` };
    }
  }
  return { ok: true };
}
