"use client";

import { useEffect, useState } from "react";
import { Check, Loader2, RefreshCw, Save, Zap } from "lucide-react";
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

interface DhanStatus {
  present: boolean;
  exp: number | null;
  expired: boolean;
  hoursLeft: number | null;
  autoRefresh: boolean;
}

/** NSE data source token (Dhan). Self-refreshes via TOTP when configured; manual paste / bot /access as fallback. */
export function DhanSettings() {
  const { user } = useAuth();
  const supabase = getSupabaseBrowser();
  const [token, setToken] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [status, setStatus] = useState<DhanStatus | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadStatus = () => {
    fetch("/api/dhan", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => setStatus(s))
      .catch(() => setStatus(null));
  };

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
    loadStatus();
    return () => {
      active = false;
    };
  }, [supabase, user]);

  const localExp = token ? expOf(token) : null;
  const now = Math.floor(Date.now() / 1000);
  const tokenStatus = !token
    ? null
    : localExp == null
      ? { txt: "Unrecognized token (no expiry found).", ok: false }
      : localExp <= now
        ? { txt: "Expired — NSE is on Yahoo fallback. Paste a fresh token.", ok: false }
        : { txt: `Valid — expires ${new Date(localExp * 1000).toLocaleString()} (~${Math.round((localExp - now) / 3600)}h left).`, ok: true };

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
    loadStatus();
  };

  const refreshNow = async () => {
    if (!user) return;
    setRefreshing(true);
    setMsg(null);
    try {
      const r = await fetch("/api/dhan", { method: "POST" });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) {
        setMsg(j.error ?? "Refresh failed.");
      } else {
        setMsg(`Fresh token minted — expires ${new Date((j.exp ?? 0) * 1000).toLocaleString()}.`);
        // Reflect the new token in the textarea + status.
        const { data } = await supabase.from("stocker_settings").select("dhan_token").eq("user_id", user.id).maybeSingle();
        setToken(((data?.dhan_token as string | null) ?? "").trim());
        loadStatus();
      }
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Refresh failed.");
    }
    setRefreshing(false);
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
            {status?.autoRefresh ? (
              <div className="flex items-start gap-2 rounded-lg border border-bull/30 bg-bull/10 px-3 py-2">
                <Zap className="mt-0.5 h-4 w-4 shrink-0 text-bull" />
                <div className="text-2xs leading-relaxed text-text">
                  <span className="font-semibold text-bull">Auto-refresh ON.</span> The token mints itself from your TOTP
                  secret and never needs manual pasting. It refreshes automatically when expired
                  {status.hoursLeft != null ? ` (currently ~${status.hoursLeft}h left)` : ""}.
                </div>
              </div>
            ) : (
              <p className="text-2xs leading-relaxed text-faint">
                Dhan powers NSE prices (US always uses Yahoo). The token expires roughly daily — paste a fresh one from
                DhanHQ here, or send <span className="font-mono">/access &lt;token&gt;</span> to{" "}
                <span className="font-mono">@alphastocker_bot</span>. To stop pasting daily, set the
                <span className="font-mono"> DHAN_TOTP_SECRET / DHAN_CLIENT_ID / DHAN_PIN</span> env vars for hands-free
                auto-refresh.
              </p>
            )}

            <textarea
              rows={3}
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Paste Dhan access token… (optional when auto-refresh is on)"
              autoComplete="off"
              className={cn(field, "resize-y font-mono text-xs")}
            />
            {tokenStatus && <p className={cn("text-xs", tokenStatus.ok ? "text-bull" : "text-bear")}>{tokenStatus.txt}</p>}

            <div className="flex flex-wrap items-center gap-3">
              <Button type="button" onClick={save} disabled={saving}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : saved ? <Check className="h-4 w-4" /> : <Save className="h-4 w-4" />}
                {saved ? "Saved" : "Save token"}
              </Button>
              {status?.autoRefresh && (
                <Button type="button" variant="secondary" onClick={refreshNow} disabled={refreshing}>
                  {refreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                  Refresh now
                </Button>
              )}
              {msg && <span className={cn("text-xs", msg.includes("minted") ? "text-bull" : "text-bear")}>{msg}</span>}
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}
