"use client";

import { useEffect, useState } from "react";
import { Fingerprint, Loader2 } from "lucide-react";
import { startAuthentication } from "@simplewebauthn/browser";

const ENROLLED_FLAG = "stocker-passkey-enrolled";

/**
 * "Unlock with Face ID / fingerprint" button for the login screen. Shown only on
 * devices that support WebAuthn AND have previously enrolled a passkey here. A
 * successful assertion mints a real Supabase session server-side, then we do a
 * full navigation so the middleware + client pick up the new cookies.
 */
export function BiometricUnlock() {
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      const supported = typeof window !== "undefined" && !!window.PublicKeyCredential;
      setShow(supported && localStorage.getItem(ENROLLED_FLAG) === "1");
    } catch {
      setShow(false);
    }
  }, []);

  const unlock = async () => {
    setBusy(true);
    setError(null);
    try {
      const optRes = await fetch("/api/passkey/auth/options", { method: "POST" });
      if (!optRes.ok) throw new Error("Could not start biometric unlock.");
      const options = await optRes.json();
      const assertion = await startAuthentication({ optionsJSON: options });
      const verifyRes = await fetch("/api/passkey/auth/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(assertion),
      });
      const data = await verifyRes.json();
      if (!verifyRes.ok || !data.ok) throw new Error(data.error ?? "Unlock failed.");
      window.location.assign("/");
    } catch (e) {
      // User cancelled the native prompt — not an error worth surfacing.
      if (e instanceof Error && e.name === "NotAllowedError") setError(null);
      else setError(e instanceof Error ? e.message : "Unlock failed.");
      setBusy(false);
    }
  };

  if (!show) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 py-1">
        <span className="h-px flex-1 bg-border" />
        <span className="text-2xs uppercase tracking-widest text-faint">or</span>
        <span className="h-px flex-1 bg-border" />
      </div>
      <button
        type="button"
        onClick={unlock}
        disabled={busy}
        className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm font-medium text-text transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60 disabled:opacity-60"
      >
        {busy ? (
          <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} />
        ) : (
          <Fingerprint className="h-4 w-4 text-brand" strokeWidth={2} />
        )}
        Unlock with Face ID / fingerprint
      </button>
      {error && <p className="text-xs text-bear">{error}</p>}
    </div>
  );
}
