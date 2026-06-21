"use client";

import { useEffect, useState } from "react";
import { Fingerprint, Loader2, ShieldCheck, Trash2 } from "lucide-react";
import { startRegistration } from "@simplewebauthn/browser";
import { Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";

const ENROLLED_FLAG = "stocker-passkey-enrolled";

/**
 * Settings card to enable/disable biometric unlock (Touch ID / Face ID / Windows
 * Hello) for this account on this device. Enrolment requires the current session;
 * after it succeeds, the login screen offers a one-tap biometric unlock.
 */
export function BiometricEnroll() {
  const [enrolled, setEnrolled] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    setSupported(typeof window !== "undefined" && !!window.PublicKeyCredential);
    fetch("/api/passkey/status")
      .then((r) => r.json())
      .then((d) => setEnrolled(!!d.enrolled))
      .catch(() => setEnrolled(false));
  }, []);

  const setFlag = (on: boolean) => {
    try {
      if (on) localStorage.setItem(ENROLLED_FLAG, "1");
      else localStorage.removeItem(ENROLLED_FLAG);
    } catch {
      /* ignore storage failures */
    }
  };

  const enroll = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const optRes = await fetch("/api/passkey/register/options", { method: "POST" });
      if (!optRes.ok) throw new Error((await optRes.json()).error ?? "Could not start.");
      const options = await optRes.json();
      const attestation = await startRegistration({ optionsJSON: options });
      const verifyRes = await fetch("/api/passkey/register/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(attestation),
      });
      const data = await verifyRes.json();
      if (!verifyRes.ok || !data.ok) throw new Error(data.error ?? "Enrolment failed.");
      setFlag(true);
      setEnrolled(true);
      setMsg({ text: "Biometric unlock enabled on this device.", ok: true });
    } catch (e) {
      if (e instanceof Error && e.name === "NotAllowedError") setMsg({ text: "Cancelled.", ok: false });
      else setMsg({ text: e instanceof Error ? e.message : "Enrolment failed.", ok: false });
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await fetch("/api/passkey/status", { method: "DELETE" });
      const d = await r.json();
      if (!r.ok || !d.ok) throw new Error(d.error ?? "Failed to remove.");
      setFlag(false);
      setEnrolled(false);
      setMsg({ text: "Biometric unlock removed.", ok: true });
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : "Failed to remove.", ok: false });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Biometric unlock</CardTitle>
      </CardHeader>
      <CardBody>
        {!supported ? (
          <p className="text-sm text-muted">
            This device or browser doesn&apos;t support biometric sign-in (Face ID / fingerprint / Windows Hello).
          </p>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-muted">
              Sign in with Face ID, fingerprint, or your device PIN instead of typing your password. The
              passkey is stored on this device; enrol on each device you want to use.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              {enrolled ? (
                <>
                  <span className="inline-flex items-center gap-1.5 rounded-lg bg-bull/10 px-2.5 py-1.5 text-xs font-medium text-bull ring-1 ring-bull/30">
                    <ShieldCheck className="h-4 w-4" strokeWidth={2} />
                    Enabled on this device
                  </span>
                  <Button type="button" variant="secondary" onClick={remove} disabled={busy}>
                    {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : <Trash2 className="h-4 w-4" strokeWidth={2} />}
                    Remove
                  </Button>
                  <Button type="button" variant="secondary" onClick={enroll} disabled={busy}>
                    <Fingerprint className="h-4 w-4" strokeWidth={2} />
                    Add another device
                  </Button>
                </>
              ) : (
                <Button type="button" onClick={enroll} disabled={busy || enrolled === null}>
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : <Fingerprint className="h-4 w-4" strokeWidth={2} />}
                  Enable biometric unlock
                </Button>
              )}
            </div>
            {msg && <p className={`text-xs ${msg.ok ? "text-bull" : "text-bear"}`}>{msg.text}</p>}
          </div>
        )}
      </CardBody>
    </Card>
  );
}
