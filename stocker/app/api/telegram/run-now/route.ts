import { NextResponse } from "next/server";
import { createSupabaseServer } from "@/lib/supabase/server";
import { isAdminEmail } from "@/lib/server/admins";
import { sendSetupDigest } from "@/lib/server/digest";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

/** In-app "Run now" — admin-gated by the session. ?mode=preclose for the provisional run. */
export async function POST(req: Request) {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user || !isAdminEmail(user.email)) return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const provisional = new URL(req.url).searchParams.get("mode") === "preclose";
  const result = await sendSetupDigest({ provisional, origin: new URL(req.url).origin, force: true });
  return NextResponse.json(result);
}
