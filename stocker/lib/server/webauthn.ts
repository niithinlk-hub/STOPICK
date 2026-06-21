/**
 * Relying-party identity for WebAuthn, derived from the incoming request so the
 * same code works on localhost and on the Vercel domain without configuration.
 * - rpID is the bare hostname (no port) — e.g. "localhost" or "stockeralpha.vercel.app".
 * - origin must exactly match the browser's window.location.origin.
 */
export const RP_NAME = "Stocker Analytics";

export function getRP(req: Request) {
  const h = req.headers;
  const host = h.get("x-forwarded-host") ?? h.get("host") ?? new URL(req.url).host;
  let proto = h.get("x-forwarded-proto") ?? "";
  if (!proto) proto = new URL(req.url).protocol.replace(":", "") || "https";
  const hostname = host.split(":")[0];
  return { rpID: hostname, origin: `${proto}://${host}`, rpName: RP_NAME };
}

export const REG_CHALLENGE_COOKIE = "wa_reg_chal";
export const AUTH_CHALLENGE_COOKIE = "wa_auth_chal";

export const challengeCookieOptions = {
  httpOnly: true as const,
  sameSite: "lax" as const,
  secure: process.env.NODE_ENV === "production",
  path: "/",
  maxAge: 300,
};
