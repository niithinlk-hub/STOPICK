import { NextResponse } from "next/server";
import { createSupabaseServer } from "@/lib/supabase/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Whether the signed-in user has any passkey enrolled (owner-scoped via RLS). */
export async function GET() {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ enrolled: false, count: 0 });

  const { count } = await supabase
    .from("stocker_passkeys")
    .select("cred_id", { count: "exact", head: true })
    .eq("user_id", user.id);

  return NextResponse.json({ enrolled: (count ?? 0) > 0, count: count ?? 0 });
}

/** Revoke all of the signed-in user's passkeys. */
export async function DELETE() {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });

  const { error } = await supabase.from("stocker_passkeys").delete().eq("user_id", user.id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true });
}
