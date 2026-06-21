"use client";

import { useEffect, useState } from "react";
import { Check, Eye, EyeOff, Loader2, Save } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";
import { getSupabaseBrowser } from "@/lib/supabase/client";
import { useAuth } from "@/components/auth/AuthProvider";
import { SignInNotice } from "@/components/auth/SignInNotice";
import { BiometricEnroll } from "@/components/auth/BiometricEnroll";
import { UserManagement } from "@/components/admin/UserManagement";

const DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai";
const DEFAULT_MODEL = "gemini-2.0-flash";

export function AdminSettingsClient() {
  const { user, loading: authLoading } = useAuth();
  const supabase = getSupabaseBrowser();

  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [apiKey, setApiKey] = useState("");
  const [newsEnabled, setNewsEnabled] = useState(true);
  const [showKey, setShowKey] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [newPassword, setNewPassword] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [pwMsg, setPwMsg] = useState<string | null>(null);

  const updatePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword.length < 6) {
      setPwMsg("Password must be at least 6 characters.");
      return;
    }
    setPwBusy(true);
    setPwMsg(null);
    const { error } = await supabase.auth.updateUser({ password: newPassword });
    setPwBusy(false);
    setPwMsg(error ? error.message : "Password updated.");
    if (!error) setNewPassword("");
  };

  useEffect(() => {
    if (!user) {
      setLoaded(true);
      return;
    }
    let active = true;
    supabase
      .from("stocker_settings")
      .select("*")
      .eq("user_id", user.id)
      .maybeSingle()
      .then(({ data }) => {
        if (!active) return;
        if (data) {
          setBaseUrl(data.llm_base_url ?? DEFAULT_BASE);
          setModel(data.llm_model ?? DEFAULT_MODEL);
          setApiKey(data.llm_api_key ?? "");
          setNewsEnabled(data.news_enabled ?? true);
        }
        setLoaded(true);
      });
    return () => {
      active = false;
    };
  }, [supabase, user]);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user) return;
    setSaving(true);
    setError(null);
    const { error } = await supabase.from("stocker_settings").upsert(
      {
        user_id: user.id,
        llm_base_url: baseUrl.trim() || null,
        llm_model: model.trim() || null,
        llm_api_key: apiKey.trim() || null,
        news_enabled: newsEnabled,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "user_id" },
    );
    setSaving(false);
    if (error) {
      setError(error.message);
    } else {
      setSaved(true);
      window.setTimeout(() => setSaved(false), 2200);
    }
  };

  const field =
    "w-full rounded-lg border border-border bg-elevated/60 px-3 py-2 font-mono text-sm text-text placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";
  const lbl = "text-2xs font-medium uppercase tracking-widest text-faint";

  return (
    <div className="mx-auto max-w-[760px] px-4 py-5 sm:px-6">
      <header className="mb-5">
        <h1 className="text-lg font-semibold tracking-tight">Settings</h1>
        <p className="mt-0.5 text-sm text-muted">
          Connect any OpenAI-compatible AI model used to score commodity news. The key is
          private to your account.
        </p>
      </header>

      {!authLoading && !user ? (
        <SignInNotice feature="settings" />
      ) : !loaded ? (
        <Card>
          <CardBody className="py-12 text-center text-sm text-muted">Loading…</CardBody>
        </Card>
      ) : (
        <div className="space-y-5">
        <Card>
          <CardHeader>
            <CardTitle>News LLM (OpenAI-compatible)</CardTitle>
          </CardHeader>
          <CardBody>
            <form onSubmit={save} className="space-y-4">
              <div className="space-y-1.5">
                <label htmlFor="base" className={lbl}>
                  Base URL
                </label>
                <input id="base" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder={DEFAULT_BASE} className={field} />
                <p className="text-2xs text-faint">
                  Default endpoint: <span className="font-mono">{DEFAULT_BASE}</span> (chat/completions is appended).
                </p>
              </div>

              <div className="space-y-1.5">
                <label htmlFor="model" className={lbl}>
                  Model
                </label>
                <input id="model" value={model} onChange={(e) => setModel(e.target.value)} placeholder={DEFAULT_MODEL} className={field} />
                <p className="text-2xs text-faint">e.g. gpt-4o-mini, or your provider&apos;s model id.</p>
              </div>

              <div className="space-y-1.5">
                <label htmlFor="key" className={lbl}>
                  API Key
                </label>
                <div className="relative">
                  <input
                    id="key"
                    type={showKey ? "text" : "password"}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="AIza… / sk-…"
                    autoComplete="off"
                    className={cn(field, "pr-10")}
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey((v) => !v)}
                    aria-label={showKey ? "Hide key" : "Show key"}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-faint hover:text-text"
                  >
                    {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              <label className="flex items-center gap-2.5 text-sm text-text">
                <input
                  type="checkbox"
                  checked={newsEnabled}
                  onChange={(e) => setNewsEnabled(e.target.checked)}
                  className="h-4 w-4 rounded border-border bg-elevated accent-brand"
                />
                Fuse live news sentiment into commodity predictions
              </label>

              {error && <p className="text-xs text-bear">{error}</p>}

              <div className="flex items-center gap-3">
                <Button type="submit" disabled={saving}>
                  {saving ? (
                    <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} />
                  ) : saved ? (
                    <Check className="h-4 w-4" strokeWidth={2} />
                  ) : (
                    <Save className="h-4 w-4" strokeWidth={2} />
                  )}
                  {saved ? "Saved" : "Save settings"}
                </Button>
                <span className="text-2xs text-faint">Stored privately (row-level security). Used server-side only.</span>
              </div>
            </form>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Account password</CardTitle>
          </CardHeader>
          <CardBody>
            <form onSubmit={updatePassword} className="flex flex-wrap items-end gap-3">
              <div className="flex-1 space-y-1.5" style={{ minWidth: "16rem" }}>
                <label htmlFor="newpw" className={lbl}>
                  New password
                </label>
                <input
                  id="newpw"
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="At least 6 characters"
                  autoComplete="new-password"
                  className={field}
                />
              </div>
              <Button type="submit" variant="secondary" disabled={pwBusy || newPassword.length < 6}>
                {pwBusy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : null}
                Update password
              </Button>
              {pwMsg && (
                <p className={cn("w-full text-xs", pwMsg === "Password updated." ? "text-bull" : "text-bear")}>{pwMsg}</p>
              )}
            </form>
          </CardBody>
        </Card>

        <BiometricEnroll />

        <UserManagement currentUserId={user?.id} />
        </div>
      )}
    </div>
  );
}
