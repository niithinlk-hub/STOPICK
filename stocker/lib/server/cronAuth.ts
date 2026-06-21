/**
 * Cron route guard. Vercel attaches `Authorization: Bearer <CRON_SECRET>` to cron
 * invocations when CRON_SECRET is set; the same secret authorizes manual/external
 * triggers (e.g. cron-job.org for intraday). Open only when no secret is configured.
 */
export function cronAuthorized(req: Request): boolean {
  const secret = process.env.CRON_SECRET;
  if (!secret) return true;
  return req.headers.get("authorization") === `Bearer ${secret}`;
}
