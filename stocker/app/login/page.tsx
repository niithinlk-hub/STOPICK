"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Check, Loader2, LogIn, Mail } from "lucide-react";
import { Button, Card, CardBody } from "@/components/ui/primitives";
import { StockerLogo } from "@/components/brand/StockerLogo";
import { BiometricUnlock } from "@/components/auth/BiometricUnlock";
import { getSupabaseBrowser } from "@/lib/supabase/client";

// Bare usernames (no "@") map to this domain, so "admin" works as a login.
const ADMIN_DOMAIN = "stocker.app";
const toEmail = (id: string) => (id.includes("@") ? id : `${id.trim()}@${ADMIN_DOMAIN}`);

export default function LoginPage() {
  const supabase = getSupabaseBrowser();
  const router = useRouter();
  const [mode, setMode] = useState<"signin" | "reset">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const signIn = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password) return;
    setBusy(true);
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email: toEmail(email), password });
    if (error) {
      setError(error.message);
      setBusy(false);
    } else {
      router.push("/");
      router.refresh();
    }
  };

  const sendReset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    setBusy(true);
    setError(null);
    const { error } = await supabase.auth.resetPasswordForEmail(toEmail(email), {
      redirectTo: `${window.location.origin}/auth/callback?next=/reset-password`,
    });
    setBusy(false);
    if (error) setError(error.message);
    else setSent(true);
  };

  const field =
    "w-full rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm text-text placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";
  const lbl = "text-2xs font-medium uppercase tracking-widest text-faint";

  return (
    <div className="mx-auto flex min-h-dvh max-w-md flex-col justify-center px-4 py-10">
      <div className="mb-7 flex flex-col items-center text-center">
        <StockerLogo size={72} />
        <div className="mt-3 text-lg font-bold tracking-tight text-text">STOCKER</div>
        <div className="text-2xs uppercase tracking-[0.3em] text-faint">Analytics</div>
      </div>

      <Card>
        <CardBody className="space-y-4">
          {mode === "reset" ? (
            sent ? (
              <div className="flex flex-col items-center gap-3 py-6 text-center">
                <span className="grid h-12 w-12 place-items-center rounded-2xl bg-bull/10 ring-1 ring-bull/30">
                  <Check className="h-6 w-6 text-bull" strokeWidth={2} />
                </span>
                <p className="text-sm text-text">Reset link sent</p>
                <p className="text-xs text-muted">
                  Check the inbox for <span className="font-medium text-text">{toEmail(email)}</span> and open the link to set a new password.
                </p>
                <button type="button" onClick={() => { setMode("signin"); setSent(false); }} className="text-2xs text-brand hover:underline">
                  Back to sign in
                </button>
              </div>
            ) : (
              <form onSubmit={sendReset} className="space-y-3">
                <div className="space-y-1.5">
                  <label htmlFor="email" className={lbl}>Email</label>
                  <input id="email" type="text" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@email.com" className={field} />
                </div>
                <Button type="submit" className="w-full" disabled={busy || !email.trim()}>
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : <Mail className="h-4 w-4" strokeWidth={2} />}
                  Email me a reset link
                </Button>
                {error && <p className="text-xs text-bear">{error}</p>}
                <button type="button" onClick={() => { setMode("signin"); setError(null); }} className="text-2xs text-muted hover:text-text">
                  ← Back to sign in
                </button>
              </form>
            )
          ) : (
            <>
            <form onSubmit={signIn} className="space-y-3">
              <div className="space-y-1.5">
                <label htmlFor="email" className={lbl}>Email</label>
                <input id="email" type="text" required autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@email.com" className={field} />
              </div>
              <div className="space-y-1.5">
                <label htmlFor="password" className={lbl}>Password</label>
                <input id="password" type="password" required autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" className={field} />
              </div>
              <Button type="submit" className="w-full" disabled={busy || !email.trim() || !password}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : <LogIn className="h-4 w-4" strokeWidth={2} />}
                Sign in
              </Button>
              {error && <p className="text-xs text-bear">{error}</p>}
              <div className="flex items-center justify-between text-2xs">
                <span className="text-faint">Private to your account.</span>
                <button type="button" onClick={() => { setMode("reset"); setError(null); }} className="text-brand hover:underline">
                  Forgot password?
                </button>
              </div>
            </form>
            <BiometricUnlock />
            </>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
