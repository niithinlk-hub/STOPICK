import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { verifyRegistrationResponse } from "@simplewebauthn/server";
import { isoBase64URL } from "@simplewebauthn/server/helpers";
import { createSupabaseServer } from "@/lib/supabase/server";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { getRP, REG_CHALLENGE_COOKIE } from "@/lib/server/webauthn";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Finish passkey enrolment: verify the attestation and store the credential. */
export async function POST(req: Request) {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });

  const body = await req.json();
  const expectedChallenge = cookies().get(REG_CHALLENGE_COOKIE)?.value;
  if (!expectedChallenge) return NextResponse.json({ error: "Challenge expired — try again." }, { status: 400 });

  const { rpID, origin } = getRP(req);
  let verification;
  try {
    verification = await verifyRegistrationResponse({
      response: body,
      expectedChallenge,
      expectedOrigin: origin,
      expectedRPID: rpID,
      requireUserVerification: true,
    });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : "Verification failed" }, { status: 400 });
  }
  cookies().delete(REG_CHALLENGE_COOKIE);

  if (!verification.verified || !verification.registrationInfo) {
    return NextResponse.json({ error: "Could not verify the passkey." }, { status: 400 });
  }

  const { credential, credentialDeviceType } = verification.registrationInfo;
  const admin = getSupabaseAdmin();
  const { error } = await admin.from("stocker_passkeys").upsert(
    {
      cred_id: credential.id,
      user_id: user.id,
      public_key: isoBase64URL.fromBuffer(credential.publicKey),
      counter: credential.counter,
      transports: credential.transports ?? [],
      device_label: credentialDeviceType === "multiDevice" ? "Synced passkey" : "This device",
    },
    { onConflict: "cred_id" },
  );
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true });
}
