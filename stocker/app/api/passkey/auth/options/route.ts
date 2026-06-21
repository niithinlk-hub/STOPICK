import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { generateAuthenticationOptions } from "@simplewebauthn/server";
import { getRP, AUTH_CHALLENGE_COOKIE, challengeCookieOptions } from "@/lib/server/webauthn";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Begin biometric unlock — no session required. Uses discoverable credentials. */
export async function POST(req: Request) {
  const { rpID } = getRP(req);
  const options = await generateAuthenticationOptions({
    rpID,
    userVerification: "required",
    allowCredentials: [],
  });
  cookies().set(AUTH_CHALLENGE_COOKIE, options.challenge, challengeCookieOptions);
  return NextResponse.json(options);
}
