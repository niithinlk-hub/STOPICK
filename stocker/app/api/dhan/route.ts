import { NextResponse } from "next/server";
import { createSupabaseServer } from "@/lib/supabase/server";
import { isAdminEmail } from "@/lib/server/admins";
import { getDhanTokenStatus, refreshDhanToken } from "@/lib/server/dhanToken";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function requireAdmin() {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user || !isAdminEmail(user.email)) return null;
  return user;
}

/** Current Dhan token health + whether TOTP auto-refresh is configured. */
export async function GET() {
  if (!(await requireAdmin())) return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  return NextResponse.json(await getDhanTokenStatus());
}

/** Force-mint a fresh 24h token via TOTP now. */
export async function POST() {
  if (!(await requireAdmin())) return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  const r = await refreshDhanToken();
  if (!r.ok) return NextResponse.json({ error: r.error ?? "Refresh failed." }, { status: 400 });
  return NextResponse.json({ ok: true, exp: r.exp });
}
