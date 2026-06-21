import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { generateRegistrationOptions } from "@simplewebauthn/server";
import { createSupabaseServer } from "@/lib/supabase/server";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { getRP, REG_CHALLENGE_COOKIE, challengeCookieOptions } from "@/lib/server/webauthn";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Begin passkey enrolment — requires an existing (password) session. */
export async function POST(req: Request) {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });

  const { rpID, rpName } = getRP(req);
  const admin = getSupabaseAdmin();
  const { data: existing } = await admin
    .from("stocker_passkeys")
    .select("cred_id, transports")
    .eq("user_id", user.id);

  const options = await generateRegistrationOptions({
    rpName,
    rpID,
    userName: user.email ?? user.id,
    userID: new TextEncoder().encode(user.id),
    attestationType: "none",
    excludeCredentials: (existing ?? []).map((c) => ({
      id: c.cred_id as string,
      transports: (c.transports as AuthenticatorTransport[] | null) ?? undefined,
    })),
    authenticatorSelection: {
      residentKey: "required",
      userVerification: "required",
      authenticatorAttachment: "platform",
    },
  });

  cookies().set(REG_CHALLENGE_COOKIE, options.challenge, challengeCookieOptions);
  return NextResponse.json(options);
}
