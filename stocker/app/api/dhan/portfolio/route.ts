import { NextResponse } from "next/server";
import { createSupabaseServer } from "@/lib/supabase/server";
import { isAdminEmail } from "@/lib/server/admins";
import { dhanGet } from "@/lib/data/dhan";
import { getDhanTokenStatus } from "@/lib/server/dhanToken";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function requireAdmin() {
  const {
    data: { user },
  } = await createSupabaseServer().auth.getUser();
  return user && isAdminEmail(user.email) ? user : null;
}

/**
 * Read-only Dhan portfolio snapshot: funds, positions, holdings, open orders. All are
 * GET endpoints that need no static IP. Returns each section (or null on failure) plus
 * the token status so the UI can prompt a refresh when expired.
 */
export async function GET() {
  if (!(await requireAdmin())) return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const token = await getDhanTokenStatus();
  if (!token.present) {
    return NextResponse.json({ error: "No Dhan token — set one in Settings or via /access.", token });
  }

  const [funds, positions, holdings, orders] = await Promise.all([
    dhanGet<unknown>("fundlimit"),
    dhanGet<unknown>("positions"),
    dhanGet<unknown>("holdings"),
    dhanGet<unknown>("orders"),
  ]);

  return NextResponse.json({
    token,
    funds: funds ?? null,
    positions: Array.isArray(positions) ? positions : [],
    holdings: Array.isArray(holdings) ? holdings : [],
    orders: Array.isArray(orders) ? orders : [],
  });
}
