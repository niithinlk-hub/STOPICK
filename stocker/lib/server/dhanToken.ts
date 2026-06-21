import "server-only";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { resolveAdminUserId } from "@/lib/server/telegramSettings";

/**
 * Resolves the active Dhan access token: the admin user's stored token (set via the
 * app or the bot's /access command) overrides the DHAN_ACCESS_TOKEN env var. Cached
 * in-memory briefly so the hot OHLCV path doesn't hit the DB on every call.
 */
let cache: { token: string | null; at: number } | null = null;
const TTL = 3 * 60 * 1000;

export function clearDhanTokenCache() {
  cache = null;
}

export async function getDhanToken(): Promise<string | null> {
  const now = Date.now();
  if (cache && now - cache.at < TTL) return cache.token;
  let token: string | null = process.env.DHAN_ACCESS_TOKEN ?? null;
  try {
    const sb = getSupabaseAdmin();
    const userId = await resolveAdminUserId(sb);
    if (userId) {
      const { data } = await sb.from("stocker_settings").select("dhan_token").eq("user_id", userId).maybeSingle();
      const stored = (data?.dhan_token ?? "").trim();
      if (stored) token = stored;
    }
  } catch {
    /* keep env fallback */
  }
  cache = { token, at: now };
  return token;
}

/** Decode the `exp` (epoch seconds) from a Dhan JWT, or null if not parseable. */
export function decodeExp(token: string | null): number | null {
  if (!token) return null;
  try {
    const part = token.split(".")[1];
    const payload = JSON.parse(Buffer.from(part, "base64").toString("utf8"));
    return typeof payload.exp === "number" ? payload.exp : null;
  } catch {
    return null;
  }
}

export interface DhanTokenStatus {
  present: boolean;
  exp: number | null;
  expired: boolean;
  hoursLeft: number | null;
}

export async function getDhanTokenStatus(): Promise<DhanTokenStatus> {
  const token = await getDhanToken();
  const exp = decodeExp(token);
  const now = Math.floor(Date.now() / 1000);
  return {
    present: Boolean(token),
    exp,
    expired: exp != null ? exp <= now : !token,
    hoursLeft: exp != null ? Math.round((exp - now) / 3600) : null,
  };
}

/** Persist a new Dhan token to the admin user's settings (used by /access). */
export async function storeDhanToken(token: string): Promise<{ ok: boolean; exp: number | null; error?: string }> {
  const clean = token.trim();
  const exp = decodeExp(clean);
  if (!exp) return { ok: false, exp: null, error: "That doesn't look like a valid Dhan token (no expiry found)." };
  try {
    const sb = getSupabaseAdmin();
    const userId = await resolveAdminUserId(sb);
    if (!userId) return { ok: false, exp, error: "Admin user not found." };
    const { error } = await sb
      .from("stocker_settings")
      .upsert({ user_id: userId, dhan_token: clean, updated_at: new Date().toISOString() }, { onConflict: "user_id" });
    if (error) return { ok: false, exp, error: error.message };
    clearDhanTokenCache();
    return { ok: true, exp };
  } catch (e) {
    return { ok: false, exp, error: e instanceof Error ? e.message : "error" };
  }
}
