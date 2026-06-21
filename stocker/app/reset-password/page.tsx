"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, Loader2, Lock } from "lucide-react";
import { Button, Card, CardBody } from "@/components/ui/primitives";
import { StockerLogo } from "@/components/brand/StockerLogo";
import { getSupabaseBrowser } from "@/lib/supabase/client";

export default function ResetPasswordPage() {
  const supabase = getSupabaseBrowser();
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [hasSession, setHasSession] = useState(false);
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      setHasSession(Boolean(data.user));
      setReady(true);
    });
  }, [supabase]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    setBusy(true);
    setError(null);
    const { error } = await supabase.auth.updateUser({ password });
    setBusy(false);
    if (error) setError(error.message);
    else {
      setDone(true);
      window.setTimeout(() => router.push("/"), 1400);
    }
  };

  const field =
    "w-full rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm text-text placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";

  return (
    <div className="mx-auto flex min-h-dvh max-w-md flex-col justify-center px-4 py-10">
      <div className="mb-7 flex flex-col items-center text-center">
        <StockerLogo size={64} />
        <div className="mt-3 text-base font-bold tracking-tight text-text">Set a new password</div>
      </div>
      <Card>
        <CardBody>
          {!ready ? (
            <p className="py-6 text-center text-sm text-muted">Loading…</p>
          ) : done ? (
            <div className="flex flex-col items-center gap-3 py-6 text-center">
              <span className="grid h-12 w-12 place-items-center rounded-2xl bg-bull/10 ring-1 ring-bull/30">
                <Check className="h-6 w-6 text-bull" strokeWidth={2} />
              </span>
              <p className="text-sm text-text">Password updated — signing you in…</p>
            </div>
          ) : !hasSession ? (
            <div className="flex flex-col items-center gap-3 py-6 text-center">
              <Lock className="h-6 w-6 text-faint" strokeWidth={1.75} />
              <p className="text-sm text-muted">
                This page only works from a password-reset email link. Request one from the sign-in page.
              </p>
              <Button variant="secondary" onClick={() => router.push("/login")}>Back to sign in</Button>
            </div>
          ) : (
            <form onSubmit={submit} className="space-y-3">
              <div className="space-y-1.5">
                <label htmlFor="np" className="text-2xs font-medium uppercase tracking-widest text-faint">New password</label>
                <input id="np" type="password" autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 6 characters" className={field} />
              </div>
              <Button type="submit" className="w-full" disabled={busy || password.length < 6}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : null}
                Update password
              </Button>
              {error && <p className="text-xs text-bear">{error}</p>}
            </form>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
