"use client";

import { useState } from "react";
import { Check, FlaskConical, Loader2, Play, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";
import { useAuth } from "@/components/auth/AuthProvider";
import { SignInNotice } from "@/components/auth/SignInNotice";

interface GradeBucket {
  grade: string;
  trades: number;
  winRate: number;
  expectancyR: number;
  targetHitRate: number;
  avgFwdRetPct: number;
}
interface ScoreBin { label: string; trades: number; winRate: number; avgR: number }
interface Result {
  market: string;
  source: string;
  sampleWithData: number;
  horizon: number;
  liqWeight: number;
  totalSignals: number;
  byGrade: GradeBucket[];
  byScoreBin: ScoreBin[];
  overall: { trades: number; winRate: number; expectancyR: number; targetHitRate: number };
  monotonicGrade: boolean;
  spearman: number;
  notes: string[];
  error?: string;
}

const SETS = [
  { v: "tier_1", l: "Set 1" },
  { v: "tier_2", l: "Set 2" },
  { v: "tier_3", l: "Set 3" },
];

const rColor = (r: number) => (r > 0.15 ? "text-bull" : r < -0.05 ? "text-bear" : "text-text");

export function AlgoCheckClient() {
  const { user, loading: authLoading } = useAuth();
  const [market, setMarket] = useState<"NSE" | "US">("NSE");
  const [source, setSource] = useState("tier_1");
  const [sample, setSample] = useState(20);
  const [horizon, setHorizon] = useState(10);
  const [liqWeight, setLiqWeight] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<Result | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setErr(null);
    setRes(null);
    try {
      const qs = new URLSearchParams({ market, source, sample: String(sample), horizon: String(horizon) });
      if (liqWeight.trim()) qs.set("liqWeight", liqWeight.trim());
      const r = await fetch(`/api/algo-check?${qs}`, { cache: "no-store" });
      if (r.status === 403) {
        setErr("Algo Check is owner-only. Sign in as admin.");
        return;
      }
      const j = (await r.json()) as Result;
      if (j.error) setErr(j.error);
      else setRes(j);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Run failed.");
    } finally {
      setBusy(false);
    }
  };

  if (!authLoading && !user) {
    return (
      <div className="mx-auto max-w-[960px] px-4 py-5 sm:px-6">
        <SignInNotice feature="Algo Check" />
      </div>
    );
  }

  const field = "rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";
  const th = "px-3 py-2 text-left text-2xs font-medium uppercase tracking-widest text-faint";
  const td = "px-3 py-2 text-sm tabular-nums text-text";

  return (
    <div className="mx-auto max-w-[1000px] px-4 py-5 sm:px-6">
      <header className="mb-5">
        <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
          <FlaskConical className="h-5 w-5 text-brand" /> Algo Check
        </h1>
        <p className="mt-0.5 text-sm text-muted">
          Validates the screening engine itself — walks daily history with no lookahead, runs the real engine at each
          bar, enters next-open, and measures forward outcome to a 2R target / ATR stop. If the engine is sound, a higher
          grade should earn a higher expectancy. The live Scanner is untouched.
        </p>
      </header>

      <Card>
        <CardBody>
          <div className="flex flex-wrap items-end gap-3">
            <label className="space-y-1">
              <span className="block text-2xs uppercase tracking-widest text-faint">Market</span>
              <select value={market} onChange={(e) => setMarket(e.target.value as "NSE" | "US")} className={field}>
                <option value="NSE">NSE</option>
                <option value="US">US</option>
              </select>
            </label>
            <label className="space-y-1">
              <span className="block text-2xs uppercase tracking-widest text-faint">Set</span>
              <select value={source} onChange={(e) => setSource(e.target.value)} className={field}>
                {SETS.map((s) => (
                  <option key={s.v} value={s.v}>{s.l}</option>
                ))}
              </select>
            </label>
            <label className="space-y-1">
              <span className="block text-2xs uppercase tracking-widest text-faint">Sample (≤40)</span>
              <input type="number" min={5} max={40} value={sample} onChange={(e) => setSample(Number(e.target.value))} className={cn(field, "w-24")} />
            </label>
            <label className="space-y-1">
              <span className="block text-2xs uppercase tracking-widest text-faint">Horizon (bars)</span>
              <input type="number" min={3} max={40} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} className={cn(field, "w-24")} />
            </label>
            <label className="space-y-1">
              <span className="block text-2xs uppercase tracking-widest text-faint">Liq weight (A/B)</span>
              <input type="text" placeholder="def 12" value={liqWeight} onChange={(e) => setLiqWeight(e.target.value)} className={cn(field, "w-24")} />
            </label>
            <Button type="button" onClick={run} disabled={busy}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {busy ? "Running…" : "Run validation"}
            </Button>
          </div>
          <p className="mt-2 text-2xs text-faint">Walks ~2 yrs of daily bars per name — takes ~30–60s. Sample is capped to keep it within the time budget.</p>
          {err && <p className="mt-2 text-sm text-bear">{err}</p>}
        </CardBody>
      </Card>

      {res && (
        <div className="mt-5 space-y-5">
          {/* Verdict */}
          <Card>
            <CardHeader>
              <CardTitle>Verdict</CardTitle>
            </CardHeader>
            <CardBody className="space-y-3">
              <div className="flex flex-wrap gap-2">
                <Badge className={res.monotonicGrade ? "border-bull/40 bg-bull/10 text-bull" : "border-bear/40 bg-bear/10 text-bear"}>
                  {res.monotonicGrade ? <Check className="mr-1 h-3 w-3" /> : <X className="mr-1 h-3 w-3" />}
                  Grade ordering {res.monotonicGrade ? "holds" : "violated"}
                </Badge>
                <Badge className={res.spearman > 0.1 ? "border-bull/40 bg-bull/10 text-bull" : res.spearman < 0 ? "border-bear/40 bg-bear/10 text-bear" : ""}>
                  Score↔return corr {res.spearman}
                </Badge>
                <Badge>{res.totalSignals} signals · {res.overall.trades} trades · {res.sampleWithData} names</Badge>
                <Badge>liq weight {res.liqWeight}</Badge>
              </div>
              <p className="text-sm text-muted">
                {res.monotonicGrade && res.spearman > 0.1
                  ? "Engine is calibrated: higher grade / score → better forward expectancy on this sample."
                  : res.spearman <= 0
                    ? "⚠️ Score does not (positively) predict forward return on this sample — calibration needs work."
                    : "Mixed: directionally positive but grade buckets aren't cleanly ordered. Try a larger sample/horizon."}
              </p>
              {res.notes.map((n, i) => (
                <p key={i} className="text-xs text-faint">{n}</p>
              ))}
            </CardBody>
          </Card>

          {/* Per-grade */}
          <Card>
            <CardHeader>
              <CardTitle>By grade — does grade predict outcome?</CardTitle>
            </CardHeader>
            <CardBody className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b border-border">
                      <th className={th}>Grade</th>
                      <th className={th}>Trades</th>
                      <th className={th}>Win %</th>
                      <th className={th}>Expectancy (R)</th>
                      <th className={th}>2R hit %</th>
                      <th className={th}>Avg fwd %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {res.byGrade.map((g) => (
                      <tr key={g.grade} className="border-b border-border/60 last:border-0">
                        <td className={cn(td, "font-semibold")}>{g.grade}</td>
                        <td className={td}>{g.trades}</td>
                        <td className={td}>{g.winRate}%</td>
                        <td className={cn(td, "font-semibold", rColor(g.expectancyR))}>{g.expectancyR}</td>
                        <td className={td}>{g.targetHitRate}%</td>
                        <td className={cn(td, rColor(g.avgFwdRetPct / 5))}>{g.avgFwdRetPct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardBody>
          </Card>

          {/* Per-score-bin */}
          <Card>
            <CardHeader>
              <CardTitle>By score band</CardTitle>
            </CardHeader>
            <CardBody className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b border-border">
                      <th className={th}>Score</th>
                      <th className={th}>Trades</th>
                      <th className={th}>Win %</th>
                      <th className={th}>Avg R</th>
                    </tr>
                  </thead>
                  <tbody>
                    {res.byScoreBin.map((b) => (
                      <tr key={b.label} className="border-b border-border/60 last:border-0">
                        <td className={cn(td, "font-medium")}>{b.label}</td>
                        <td className={td}>{b.trades}</td>
                        <td className={td}>{b.winRate}%</td>
                        <td className={cn(td, "font-semibold", rColor(b.avgR))}>{b.avgR}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardBody>
          </Card>
        </div>
      )}
    </div>
  );
}
