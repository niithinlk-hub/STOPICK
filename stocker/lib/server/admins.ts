/**
 * Which accounts may manage users. Defaults to the bootstrap admins; override
 * with the ADMIN_EMAILS env var (comma-separated). Server-only.
 */
const DEFAULT_ADMINS = ["niithin.lk@gmail.com", "admin@stocker.app"];

export function adminEmails(): string[] {
  const env = process.env.ADMIN_EMAILS;
  return (env ? env.split(",") : DEFAULT_ADMINS).map((e) => e.trim().toLowerCase()).filter(Boolean);
}

export function isAdminEmail(email: string | null | undefined): boolean {
  if (!email) return false;
  return adminEmails().includes(email.toLowerCase());
}
