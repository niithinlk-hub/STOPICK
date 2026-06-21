"use client";

import { useEffect, useState } from "react";
import { Check, Loader2, Save, Send } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";
import { getSupabaseBrowser } from "@/lib/supabase/client";
import { useAuth } from "@/components/auth/AuthProvider";

const SETS: { key: string; label: string }[] = [
  { key: "NSE:tier_1", label: "NSE-1" },
  { key: "NSE:tier_2", label: "NSE-2" },
  { key: "NSE:tier_3", label: "NSE-3" },
  { key: "US:tier_1", label: "US-1" },
  { key: "US:tier_2", label: "US-2" },
  { key: "US:tier_3", label: "US-3" },
];
const INTRADAY_TFS = ["15m", "1h", "4h"];
const ALL_SET_KEYS = SETS.map((s) => s.key);

interface Tg {
  enabled: boolean;
  chatId: string;
  sets: string[];
  minScore: number;
  topN: number;
  preclose: boolean;
  intradayEnabled: boolean;
  intradayTf: string;
}

const DEFAULTS: Tg = {
  enabled: true,
  chatId: "",
  sets: ALL_SET_KEYS,
  minScore: 75,
  topN: 40,
  preclose: true,
  intradayEnabled: true,
  intradayTf: "15m",
};

export function TelegramSettings() {
  const { user } = useAuth();
  const supabase = getSupabaseBrowser();

  const [tg, setTg] = useState<Tg>(DEFAULTS);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [running, setRunning] = useState<"" | "run" | "preclose">("");
  const [runMsg, setRunMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!user) return;
    let active = true;
    supabase
      .from("stocker_settings")
      .select("telegram")
      .eq("user_id", user.id)
      .maybeSingle()
      .then(({ data }) => {
        if (!active) return;
        const t = (data?.telegram ?? {}) as Partial<Tg>;
        setTg({ ...DEFAULTS, ...t, sets: Array.isArray(t.sets) && t.sets.length ? t.sets : DEFAULTS.sets });
        setLoaded(true);
      });
    return () => {
      active = false;
    };
  }, [supabase, user]);

  const toggleSet = (key: string) =>
    setTg((p) => ({ ...p, sets: p.sets.includes(key) ? p.sets.filter((k) => k !== key) : [...p.sets, key] }));

  const save = async () => {
    if (!user) return;
    setSaving(true);
    const payload: Tg = { ...tg, sets: ALL_SET_KEYS.filter((k) => tg.sets.includes(k)) };
    await supabase
      .from("stocker_settings")
      .upsert({ user_id: user.id, telegram: payload, updated_at: new Date().toISOString() }, { onConflict: "user_id" });
    setSaving(false);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  };

  const runNow = async (mode: "run" | "preclose") => {
    setRunning(mode);
    setRunMsg(null);
    try {
      const r = await fetch(`/api/telegram/run-now${mode === "preclose" ? "?mode=preclose" : ""}`, { method: "POST" });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error ?? "Failed");
      setRunMsg(
        d.sent
          ? `Sent — ${d.qualified} setup(s) across ${d.scanned} scanned.`
          : `Not sent: ${d.error ?? "unknown"}`,
      );
    } catch (err) {
      setRunMsg(err instanceof Error ? err.message : "Failed.");
    } finally {
      setRunning("");
    }
  };

  const field =
    "w-full rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm text-text placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";
  const lbl = "text-2xs font-medium uppercase tracking-widest text-faint";
  const Toggle = ({ on, onClick, label }: { on: boolean; onClick: () => void; label: string }) => (
    <label className="flex cursor-pointer items-center gap-2.5 text-sm text-text">
      <input type="checkbox" checked={on} onChange={onClick} className="h-4 w-4 rounded border-border bg-elevated accent-brand" />
      {label}
    </label>
  );

  if (!user) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Telegram alerts</CardTitle>
      </CardHeader>
      <CardBody className="space-y-5">
        {!loaded ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : (
          <>
            <Toggle on={tg.enabled} onClick={() => setTg((p) => ({ ...p, enabled: !p.enabled }))} label="Daily digest enabled" />

            <div className="space-y-1.5">
              <span className={lbl}>Universe sets to scan</span>
              <div className="flex flex-wrap gap-2">
                {SETS.map((s) => {
                  const on = tg.sets.includes(s.key);
                  return (
                    <button
                      key={s.key}
                      type="button"
                      onClick={() => toggleSet(s.key)}
                      aria-pressed={on}
                      className={cn(
                        "rounded-lg px-3 py-1.5 text-xs font-medium ring-1 ring-inset transition-colors",
                        on ? "bg-brand/15 text-text ring-brand/40" : "bg-elevated/60 text-muted ring-border hover:text-text",
                      )}
                    >
                      {s.label}
                    </button>
                  );
                })}
              </div>
              <p className="text-2xs text-faint">More sets = broader coverage but heavier per run.</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label htmlFor="tg-min" className={lbl}>Min score</label>
                <input id="tg-min" type="number" min={0} max={100} value={tg.minScore}
                  onChange={(e) => setTg((p) => ({ ...p, minScore: Number(e.target.value) }))} className={field} />
              </div>
              <div className="space-y-1.5">
                <label htmlFor="tg-top" className={lbl}>Max names per message</label>
                <input id="tg-top" type="number" min={1} max={100} value={tg.topN}
                  onChange={(e) => setTg((p) => ({ ...p, topN: Number(e.target.value) }))} className={field} />
              </div>
            </div>

            <div className="space-y-1.5">
              <label htmlFor="tg-chat" className={lbl}>Recipient chat ID</label>
              <input id="tg-chat" value={tg.chatId} onChange={(e) => setTg((p) => ({ ...p, chatId: e.target.value.trim() }))}
                placeholder="defaults to the configured chat" className={cn(field, "font-mono")} />
            </div>

            <Toggle on={tg.preclose} onClick={() => setTg((p) => ({ ...p, preclose: !p.preclose }))} label="Send pre-close (provisional) run ~3 PM IST" />

            <div className="flex flex-wrap items-end gap-4">
              <Toggle on={tg.intradayEnabled} onClick={() => setTg((p) => ({ ...p, intradayEnabled: !p.intradayEnabled }))} label="Intraday watchlist alerts" />
              <div className="space-y-1.5">
                <label htmlFor="tg-tf" className={lbl}>Intraday interval</label>
                <select id="tg-tf" value={tg.intradayTf} onChange={(e) => setTg((p) => ({ ...p, intradayTf: e.target.value }))} className={field}>
                  {INTRADAY_TFS.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 border-t border-border pt-4">
              <Button type="button" onClick={save} disabled={saving}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : saved ? <Check className="h-4 w-4" /> : <Save className="h-4 w-4" />}
                {saved ? "Saved" : "Save"}
              </Button>
              <Button type="button" variant="secondary" onClick={() => runNow("run")} disabled={running !== ""}>
                {running === "run" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                Run now
              </Button>
              <Button type="button" variant="ghost" onClick={() => runNow("preclose")} disabled={running !== ""}>
                {running === "preclose" ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Run pre-close
              </Button>
              {runMsg && <span className="text-2xs text-muted">{runMsg}</span>}
            </div>

            <p className="text-2xs leading-relaxed text-faint">
              Bot: <span className="font-mono">@alphastocker_bot</span>. Send <span className="font-mono">/run</span> anytime for an
              on-demand scan. Daily schedule (EOD ~4 PM IST, pre-close ~3 PM IST) is fixed by the deploy plan; intraday alerts need
              an external ~20-min trigger (cron-job.org) during market hours.
            </p>
          </>
        )}
      </CardBody>
    </Card>
  );
}
