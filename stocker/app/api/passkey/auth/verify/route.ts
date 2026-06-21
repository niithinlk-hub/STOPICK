import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { verifyAuthenticationResponse } from "@simplewebauthn/server";
import { isoBase64URL } from "@simplewebauthn/server/helpers";
import { createSupabaseServer } from "@/lib/supabase/server";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { getRP, AUTH_CHALLENGE_COOKIE } from "@/lib/server/webauthn";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Finish biometric unlock: verify the assertion, then mint a real Supabase
 * session for the credential's owner (admin generateLink → verifyOtp sets the
 * auth cookies via the SSR client).
 */
export async function POST(req: Request) {
  const body = await req.json();
  const expectedChallenge = cookies().get(AUTH_CHALLENGE_COOKIE)?.value;
  if (!expectedChallenge) return NextResponse.json({ error: "Challenge expired — try again." }, { status: 400 });

  const credId: string | undefined = body?.id;
  if (!credId) return NextResponse.json({ error: "Malformed authentication response." }, { status: 400 });

  const admin = getSupabaseAdmin();
  const { data: row } = await admin.from("stocker_passkeys").select("*").eq("cred_id", credId).maybeSingle();
  if (!row) return NextResponse.json({ error: "This passkey isn't registered." }, { status: 400 });

  const { rpID, origin } = getRP(req);
  let verification;
  try {
    verification = await verifyAuthenticationResponse({
      response: body,
      expectedChallenge,
      expectedOrigin: origin,
      expectedRPID: rpID,
      requireUserVerification: true,
      credential: {
        id: row.cred_id as string,
        publicKey: isoBase64URL.toBuffer(row.public_key as string),
        counter: Number(row.counter),
        transports: (row.transports as AuthenticatorTransport[] | null) ?? undefined,
      },
    });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : "Verification failed" }, { status: 400 });
  }
  cookies().delete(AUTH_CHALLENGE_COOKIE);

  if (!verification.verified) return NextResponse.json({ error: "Could not verify." }, { status: 400 });

  await admin
    .from("stocker_passkeys")
    .update({ counter: verification.authenticationInfo.newCounter, last_used_at: new Date().toISOString() })
    .eq("cred_id", row.cred_id);

  const { data: u } = await admin.auth.admin.getUserById(row.user_id as string);
  const email = u.user?.email;
  if (!email) return NextResponse.json({ error: "Account not found." }, { status: 400 });

  const { data: link, error: linkErr } = await admin.auth.admin.generateLink({ type: "magiclink", email });
  const tokenHash = link?.properties?.hashed_token;
  if (linkErr || !tokenHash) {
    return NextResponse.json({ error: linkErr?.message ?? "Could not start a session." }, { status: 500 });
  }

  const supabase = createSupabaseServer();
  const { error: otpErr } = await supabase.auth.verifyOtp({ type: "magiclink", token_hash: tokenHash });
  if (otpErr) return NextResponse.json({ error: otpErr.message }, { status: 500 });

  return NextResponse.json({ ok: true });
}
