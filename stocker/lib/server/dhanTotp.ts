import "server-only";
import crypto from "node:crypto";

/**
 * Self-minting Dhan access tokens. Dhan caps token validity at 24h (SEBI rule), so
 * instead of pasting one daily we generate a fresh token on demand from a TOTP secret.
 *
 * One-time setup on web.dhan.co → My Profile → Access DhanHQ APIs → Setup TOTP. Save the
 * base32 secret behind the QR, then set three server-only env vars:
 *   DHAN_TOTP_SECRET  – the base32 secret string
 *   DHAN_CLIENT_ID    – your dhanClientId
 *   DHAN_PIN          – your Dhan login PIN
 * These three together can mint tokens for the account — treat them like a password.
 */

const AUTH_URL = "https://auth.dhan.co/app/generateAccessToken";

/** RFC 4648 base32 decode (no padding required). */
function base32Decode(input: string): Buffer {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
  const clean = input.replace(/=+$/g, "").replace(/\s+/g, "").toUpperCase();
  let bits = 0;
  let value = 0;
  const out: number[] = [];
  for (const ch of clean) {
    const idx = alphabet.indexOf(ch);
    if (idx === -1) continue;
    value = (value << 5) | idx;
    bits += 5;
    if (bits >= 8) {
      out.push((value >>> (bits - 8)) & 0xff);
      bits -= 8;
    }
  }
  return Buffer.from(out);
}

/** RFC 6238 TOTP (HMAC-SHA1, 6 digits, 30s step) — same code an authenticator app shows. */
export function totp(secret: string, atMs?: number, step = 30, digits = 6): string {
  const key = base32Decode(secret);
  const counter = Math.floor((atMs ?? Date.now()) / 1000 / step);
  const buf = Buffer.alloc(8);
  buf.writeBigUInt64BE(BigInt(counter));
  const hmac = crypto.createHmac("sha1", key).update(buf).digest();
  const offset = hmac[hmac.length - 1] & 0x0f;
  const code =
    ((hmac[offset] & 0x7f) << 24) |
    ((hmac[offset + 1] & 0xff) << 16) |
    ((hmac[offset + 2] & 0xff) << 8) |
    (hmac[offset + 3] & 0xff);
  return (code % 10 ** digits).toString().padStart(digits, "0");
}

/** True when all three TOTP env vars are set so we can auto-mint. */
export function dhanTotpConfigured(): boolean {
  return Boolean(
    process.env.DHAN_TOTP_SECRET?.trim() &&
      process.env.DHAN_CLIENT_ID?.trim() &&
      process.env.DHAN_PIN?.trim(),
  );
}

/** Decode the `exp` (epoch seconds) from a Dhan JWT without importing the token module. */
function expOf(token: string): number | null {
  try {
    const payload = JSON.parse(Buffer.from(token.split(".")[1], "base64").toString("utf8"));
    return typeof payload.exp === "number" ? payload.exp : null;
  } catch {
    return null;
  }
}

export interface MintResult {
  ok: boolean;
  token?: string;
  exp?: number;
  error?: string;
}

/** Mint a fresh 24h Dhan access token via the TOTP generateAccessToken endpoint. */
export async function mintDhanToken(): Promise<MintResult> {
  const secret = process.env.DHAN_TOTP_SECRET?.trim();
  const clientId = process.env.DHAN_CLIENT_ID?.trim();
  const pin = process.env.DHAN_PIN?.trim();
  if (!secret || !clientId || !pin) {
    return { ok: false, error: "DHAN_TOTP_SECRET / DHAN_CLIENT_ID / DHAN_PIN not configured." };
  }
  const code = totp(secret);
  const url = `${AUTH_URL}?dhanClientId=${encodeURIComponent(clientId)}&pin=${encodeURIComponent(pin)}&totp=${code}`;
  try {
    const r = await fetch(url, { method: "POST", cache: "no-store" });
    const j = (await r.json().catch(() => ({}))) as Record<string, unknown>;
    if (!r.ok) {
      const detail = (j.errorMessage ?? j.message ?? JSON.stringify(j)) as string;
      return { ok: false, error: `Dhan auth ${r.status}: ${detail}` };
    }
    const token = (j.accessToken ?? j.access_token) as string | undefined;
    if (!token) return { ok: false, error: "Dhan response had no accessToken." };
    return { ok: true, token, exp: expOf(token) ?? undefined };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "mint request failed" };
  }
}
