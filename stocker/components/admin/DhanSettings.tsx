"use client";

import { useEffect, useState } from "react";
import { Check, Loader2, Save } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";
import { getSupabaseBrowser } from "@/lib/supabase/client";
import { useAuth } from "@/components/auth/AuthProvider";

function expOf(token: string): number | null {
  try {
    const part = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const p = JSON.parse(atob(part));
    return typeof p.exp === "number" ? p.exp : null;
  } catch {
    return null;
  }
}

/** NSE data source token (Dhan). Stored per user, refreshable here or via the bot's /access. */
export function DhanSettings() {
  const { user } = useAuth();
  const supabase = getSupabaseBrowser();
  const [token, setToken] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!user) return;
    let active = true;
    supabase
      .from("stocker_settings")
      .select("dhan_token")
      .eq("user_id", user.id)
      .maybeSingle()
      .then(({ data }) => {
        if (!active) return;
        setToken(((data?.dhan_token as string | null) ?? "").trim());
        setLoaded(true);
      });
    return () => {
      active = false;
    };
  }, [supabase, user]);

  const exp = token ? expOf(token) : null;
  const now = Math.floor(Date.now() / 1000);
  const status = !token
    ? null
    : exp == null
      ? { txt: "Unrecognized token (no expiry found).", ok: false }
      : exp <= now
        ? { txt: "Expired — NSE is on Yahoo fallback. Paste a fresh token.", ok: false }
        : { txt: `Valid — expires ${new Date(exp * 1000).toLocaleString()} (~${Math.round((exp - now) / 3600)}h left).`, ok: true };

  const save = async () => {
    if (!user) return;
    const clean = token.trim();
    if (clean && expOf(clean) == null) {
      setMsg("That doesn't look like a Dhan token (no expiry).");
      return;
    }
    setSaving(true);
    setMsg(null);
    const { error } = await supabase
      .from("stocker_settings")
      .upsert({ user_id: user.id, dhan_token: clean || null, updated_at: new Date().toISOString() }, { onConflict: "user_id" });
    setSaving(false);
    if (error) {
      setMsg(error.message);
      return;
    }
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  };

  if (!user) return null;
  const field =
    "w-full rounded-lg border border-border bg-elevated/60 px-3 py-2 text-text placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";

  return (
    <Card>
      <CardHeader>
        <CardTitle>NSE data — Dhan token</CardTitle>
      </CardHeader>
      <CardBody className="space-y-3">
        {!loaded ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : (
          <>
            <p className="text-2xs leading-relaxed text-faint">
              Dhan powers NSE prices (US always uses Yahoo). The token expires roughly daily — paste a fresh one from
              DhanHQ here, or send <span className="font-mono">/access &lt;token&gt;</span> to{" "}
              <span className="font-mono">@alphastocker_bot</span>. Leave empty to use Yahoo for NSE.
            </p>
            <textarea
              rows={3}
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Paste Dhan access token…"
              autoComplete="off"
              className={cn(field, "resize-y font-mono text-xs")}
            />
            {status && <p className={cn("text-xs", status.ok ? "text-bull" : "text-bear")}>{status.txt}</p>}
            <div className="flex items-center gap-3">
              <Button type="button" onClick={save} disabled={saving}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : saved ? <Check className="h-4 w-4" /> : <Save className="h-4 w-4" />}
                {saved ? "Saved" : "Save token"}
              </Button>
              {msg && <span className="text-xs text-bear">{msg}</span>}
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}
