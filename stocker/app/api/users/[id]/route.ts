import { NextResponse } from "next/server";
import { createSupabaseServer } from "@/lib/supabase/server";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { isAdminEmail } from "@/lib/server/admins";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Delete an account (admin only; cannot delete yourself). */
export async function DELETE(_req: Request, { params }: { params: { id: string } }) {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user || !isAdminEmail(user.email)) return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  if (user.id === params.id) {
    return NextResponse.json({ error: "You can't delete your own account." }, { status: 400 });
  }

  const sb = getSupabaseAdmin();
  const { error } = await sb.auth.admin.deleteUser(params.id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true });
}
