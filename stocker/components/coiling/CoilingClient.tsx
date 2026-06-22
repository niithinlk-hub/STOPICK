"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ExternalLink, Loader2, RefreshCw, Zap } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, CardBody } from "@/components/ui/primitives";
import { scanStocks } from "@/lib/client/api";
import { tradingViewUrl } from "@/lib/tradingview";
import type { Country, WatchRow } from "@/lib/engine/types";

const MARKETS: { value: Country; label: string }[] = [
  { value: "NSE", label: "NSE 🇮🇳" },
  { value: "US", label: "US 🇺🇸" },
];
const SETS: { value: "tier_1" | "tier_2" | "tier_3"; label: string }[] = [
  { value: "tier_1", label: "Set 1" },
  { value: "tier_2", label: "Set 2" },
  { value: "tier_3", label: "Set 3" },
];

const readinessTone = (r: number) =>
  r >= 85 ? "text-bull" : r >= 70 ? "text-brand" : "text-muted";

export function CoilingClient() {
  const [market, setMarket] = useState<Country>("NSE");
  const [source, setSource] = useState<"tier_1" | "tier_2" | "tier_3">("tier_1");
  const [rows, setRows] = useState<WatchRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ranAt, setRanAt] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    abort.current?.abort();
    const ac = new AbortController();
    abort.current = ac;
    setLoading(true);
    setErr(null);
    try {
      const res = await scanStocks(
        { country: market, source, timeframe: "1d", setupMode: "both", minScore: 75, includeWatch: true, limit: 250 },
        ac.signal,
      );
      setRows(res.watch ?? []);
      setRanAt(Date.now());
    } catch (e) {
      if ((e as Error).name !== "AbortError") setErr(e instanceof Error ? e.message : "Scan failed.");
    } finally {
      setLoading(false);
    }
  }, [market, source]);

  // Auto-run on first mount.
  useEffect(() => {
    run();
    return () => abort.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const th = "px-3 py-2 text-left text-2xs font-medium uppercase tracking-widest text-faint";
  const td = "px-3 py-2 text-sm text-text";

  return (
    <div className="mx-auto max-w-[1000px] px-4 py-5 sm:px-6">
      <header className="mb-4">
        <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
          <Zap className="h-5 w-5 text-brand" /> Coiling — watch to break
        </h1>
        <p className="mt-0.5 text-sm text-muted">
          Tight bases sitting just under their breakout trigger — <span className="text-text">not fired yet</span>. Set a
          buy-stop above the trigger to catch the move early. Also pushed nightly on Telegram.
        </p>
      </header>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-lg border border-border">
          {MARKETS.map((m) => (
            <button
              key={m.value}
              onClick={() => setMarket(m.value)}
              className={cn("px-3 py-1.5 text-sm", market === m.value ? "bg-brand/15 text-brand" : "text-muted hover:text-text")}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="inline-flex overflow-hidden rounded-lg border border-border">
          {SETS.map((s) => (
            <button
              key={s.value}
              onClick={() => setSource(s.value)}
              className={cn("px-3 py-1.5 text-sm", source === s.value ? "bg-brand/15 text-brand" : "text-muted hover:text-text")}
            >
              {s.label}
            </button>
          ))}
        </div>
        <Button type="button" onClick={run} disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          {loading ? "Scanning…" : "Scan"}
        </Button>
        {ranAt && !loading && <span className="text-2xs text-faint">{rows.length} coiling</span>}
      </div>

      {err && <p className="mb-3 text-sm text-bear">{err}</p>}

      <Card>
        <CardBody className="p-0">
          {loading && !rows.length ? (
            <p className="px-3 py-10 text-center text-sm text-muted">Scanning {market} {source.replace("_", " ")}…</p>
          ) : rows.length === 0 ? (
            <p className="px-3 py-10 text-center text-sm text-muted">No coiling candidates right now.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className={th}>Ticker</th>
                    <th className={cn(th, "text-right")}>Readiness</th>
                    <th className={cn(th, "text-right")}>Trigger</th>
                    <th className={cn(th, "text-right")}>Away</th>
                    <th className={cn(th, "text-right")}>Tight</th>
                    <th className={cn(th, "text-right")}>RS</th>
                    <th className={th}>Pattern</th>
                    <th className={th}></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((w) => (
                    <tr key={w.ticker} className="border-b border-border/60 last:border-0 hover:bg-elevated/40">
                      <td className={cn(td, "font-semibold")}>{w.ticker}</td>
                      <td className={cn(td, "text-right font-semibold tabular-nums", readinessTone(w.readiness))}>
                        ⚡{Math.round(w.readiness)}
                      </td>
                      <td className={cn(td, "text-right tabular-nums")}>{w.trigger.toLocaleString()}</td>
                      <td className={cn(td, "text-right tabular-nums text-muted")}>{w.distancePct.toFixed(1)}%</td>
                      <td className={cn(td, "text-right tabular-nums text-muted")}>{Math.round(w.tightness)}</td>
                      <td className={cn(td, "text-right tabular-nums text-muted")}>{w.rsScore == null ? "—" : Math.round(w.rsScore)}</td>
                      <td className={cn(td, "text-muted")}>
                        <Badge>{w.pattern}</Badge>
                      </td>
                      <td className={cn(td, "text-right")}>
                        <a
                          href={tradingViewUrl(w.ticker, w.market)}
                          target="_blank"
                          rel="noopener noreferrer"
                          title={`Open ${w.ticker} on TradingView`}
                          className="inline-flex items-center gap-1 text-xs text-brand hover:underline"
                        >
                          Chart <ExternalLink className="h-3 w-3" />
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>

      <p className="mt-3 text-2xs text-faint">
        Anticipatory — many coil and never break. The buy-stop above the trigger only fills if it actually breaks out.
      </p>
    </div>
  );
}
