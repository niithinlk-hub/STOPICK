import "server-only";
import { getSupabaseAdmin } from "@/lib/supabase/admin";

export const ALL_SETS = ["NSE:tier_1", "NSE:tier_2", "NSE:tier_3", "US:tier_1", "US:tier_2", "US:tier_3"];

export interface TelegramSettings {
  enabled: boolean;
  chatId: string | null;
  sets: string[];
  minScore: number;
  topN: number;
  preclose: boolean;
  intradayEnabled: boolean;
  intradayTf: string;
  /** Readiness threshold (0–100) for the pre-breakout "coiling primed" alert. */
  coilingMin: number;
}

function defaults(): TelegramSettings {
  return {
    enabled: true,
    chatId: process.env.TELEGRAM_CHAT_ID ?? null,
    sets: (process.env.TELEGRAM_SETS || ALL_SETS.join(",")).split(",").map((s) => s.trim()).filter(Boolean),
    minScore: Number(process.env.TELEGRAM_MIN_SCORE ?? 75),
    topN: Number(process.env.TELEGRAM_TOP_N ?? 40),
    preclose: true,
    intradayEnabled: true,
    intradayTf: process.env.TELEGRAM_INTRADAY_TF || "15m",
    coilingMin: Number(process.env.TELEGRAM_COILING_MIN ?? 95),
  };
}

type Admin = ReturnType<typeof getSupabaseAdmin>;

/** The account whose saved Telegram settings + watchlist drive the bot (first ADMIN_EMAILS match). */
export async function resolveAdminUserId(sb: Admin): Promise<string | null> {
  const emails = (process.env.ADMIN_EMAILS || "niithin.lk@gmail.com,admin@stocker.app")
    .split(",")
    .map((e) => e.trim().toLowerCase());
  const { data } = await sb.auth.admin.listUsers({ page: 1, perPage: 200 });
  const u = (data?.users ?? []).find((x) => emails.includes((x.email ?? "").toLowerCase()));
  return u?.id ?? null;
}

/** Resolve effective Telegram settings (saved JSON merged over env/defaults), plus the admin user id. */
export async function getTelegramSettings(): Promise<TelegramSettings & { userId: string | null }> {
  const d = defaults();
  try {
    const sb = getSupabaseAdmin();
    const userId = await resolveAdminUserId(sb);
    if (!userId) return { ...d, userId: null };
    const { data } = await sb.from("stocker_settings").select("telegram").eq("user_id", userId).maybeSingle();
    const t = (data?.telegram ?? {}) as Partial<TelegramSettings>;
    return {
      enabled: typeof t.enabled === "boolean" ? t.enabled : d.enabled,
      chatId: t.chatId ?? d.chatId,
      sets: Array.isArray(t.sets) && t.sets.length ? t.sets : d.sets,
      minScore: typeof t.minScore === "number" ? t.minScore : d.minScore,
      topN: typeof t.topN === "number" ? t.topN : d.topN,
      preclose: typeof t.preclose === "boolean" ? t.preclose : d.preclose,
      intradayEnabled: typeof t.intradayEnabled === "boolean" ? t.intradayEnabled : d.intradayEnabled,
      intradayTf: t.intradayTf ?? d.intradayTf,
      coilingMin: typeof t.coilingMin === "number" ? t.coilingMin : d.coilingMin,
      userId,
    };
  } catch {
    return { ...d, userId: null };
  }
}
