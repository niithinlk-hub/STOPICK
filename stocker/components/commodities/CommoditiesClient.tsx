"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AlertTriangle, Coins, Newspaper, RefreshCw, Settings } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, CardBody, ScoreMeter, Skeleton } from "@/components/ui/primitives";
import { fmtNumber, fmtPct, fmtPrice } from "@/lib/format";
import { getCommodities } from "@/lib/client/api";
import { useAuth } from "@/components/auth/AuthProvider";
import type { CommoditiesResponse, CommodityPrediction, HorizonPrediction } from "@/lib/engine/commodities";

type Tone = "bull" | "bear" | "warn" | "neutral" | "info";
const dirTone = (d: string): Tone => (d === "bullish" ? "bull" : d === "bearish" ? "bear" : "neutral");
const sentTone = (s: number): Tone => (s >= 0.15 ? "bull" : s <= -0.15 ? "bear" : "neutral");

function HorizonRow({ h }: { h: HorizonPrediction }) {
  return (
    <div className="border-t border-border/60 py-2.5 first:border-t-0">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="w-16 text-xs font-semibold text-text">{h.label}</span>
          <Badge tone={dirTone(h.direction)} className="capitalize">
            {h.direction}
          </Badge>
          <span className="text-xs text-muted">{h.action}</span>
        </div>
        <div className="flex items-center gap-2">
          <ScoreMeter score={h.confidence} className="w-24" />
          <span className="w-9 text-right font-mono text-xs font-semibold tnum text-text">{fmtNumber(h.confidence, 0)}</span>
        </div>
      </div>
      <div className="mt-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 pl-[4.5rem]">
        <span className="text-2xs text-faint">{h.pattern}</span>
        <span className="font-mono text-2xs text-faint tnum">
          RSI {fmtNumber(h.readings.rsi, 0)} · ADX {fmtNumber(h.readings.adx, 0)} · slope {fmtPct(h.readings.trendSlopePctPerWeek)}/wk
          {h.newsScore != null && (
            <>
              {" "}
              · news {h.newsScore > 0 ? "+" : ""}
              {fmtNumber(h.newsScore, 2)}
              {h.agreement === 1 ? " ✓" : h.agreement === -1 ? " ✗" : ""}
            </>
          )}
        </span>
      </div>
    </div>
  );
}

function CommodityCard({ c }: { c: CommodityPrediction }) {
  return (
    <Card>
      <CardBody className="space-y-3">
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand/10 ring-1 ring-brand/20">
              <Coins className="h-5 w-5 text-brand" strokeWidth={2} />
            </span>
            <div>
              <div className="text-sm font-bold text-text">{c.name}</div>
              <div className="font-mono text-2xs text-faint">{c.symbol}</div>
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-lg font-semibold tnum text-text">{fmtPrice(c.price)}</div>
            <div
              className={cn(
                "font-mono text-2xs tnum",
                c.dayChangePct == null ? "text-faint" : c.dayChangePct >= 0 ? "text-bull" : "text-bear",
              )}
            >
              {fmtPct(c.dayChangePct)} today
            </div>
          </div>
        </div>

        {/* Horizons */}
        <div>
          {c.horizons.map((h) => (
            <HorizonRow key={h.horizon} h={h} />
          ))}
        </div>

        {/* News */}
        {c.news ? (
          c.news.error ? (
            <div className="flex items-start gap-2 rounded-lg border border-warn/30 bg-warn/5 p-2.5 text-2xs text-warn">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={2} />
              <span>News analysis failed: {c.news.error}</span>
            </div>
          ) : (
            <div className="space-y-2 rounded-lg border border-border bg-elevated/40 p-3">
              <div className="flex items-center gap-2">
                <Newspaper className="h-3.5 w-3.5 text-faint" strokeWidth={2} />
                <Badge tone={sentTone(c.news.score)}>{c.news.label}</Badge>
                <span className="font-mono text-2xs text-faint tnum">
                  {c.news.score > 0 ? "+" : ""}
                  {fmtNumber(c.news.score, 2)}
                </span>
              </div>
              {c.news.summary && <p className="text-2xs leading-relaxed text-muted">{c.news.summary}</p>}
              {c.news.drivers.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {c.news.drivers.map((d, i) => (
                    <span key={i} className="rounded border border-border bg-surface px-1.5 py-0.5 text-2xs text-muted">
                      {d}
                    </span>
                  ))}
                </div>
              )}
              {c.news.headlines.length > 0 && (
                <ul className="space-y-1 border-t border-border/60 pt-2">
                  {c.news.headlines.slice(0, 4).map((hl, i) => (
                    <li key={i} className="truncate text-2xs text-faint">
                      • {hl.title}
                      {hl.source ? ` — ${hl.source}` : ""}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )
        ) : null}
      </CardBody>
    </Card>
  );
}

export function CommoditiesClient() {
  const { user } = useAuth();
  const [data, setData] = useState<CommoditiesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    try {
      const res = await getCommodities(ctrl.signal);
      if (!ctrl.signal.aborted) setData(res);
    } catch (err) {
      if (!ctrl.signal.aborted) setError(err instanceof Error ? err.message : "Failed to load commodities.");
    } finally {
      if (abortRef.current === ctrl) setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    return () => abortRef.current?.abort();
  }, [load]);

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-5 sm:px-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Commodities</h1>
          <p className="mt-0.5 text-sm text-muted">
            Gold, silver &amp; crude — directional prediction with a confidence score across 1D / 1W / 1M / 2M, from
            technicals fused with live news.
          </p>
        </div>
        <Button variant="secondary" onClick={load} disabled={loading} aria-label="Refresh">
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} strokeWidth={2} />
          Refresh
        </Button>
      </header>

      {/* News config hint */}
      {data && !data.newsConfigured && (
        <Card className="mb-4 border-info/30 bg-info/5">
          <CardBody className="flex flex-col items-start gap-2 py-3 text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
            <span>Showing technical-only predictions. Add an AI model key in Settings to fuse live international news.</span>
            <Link href={user ? "/admin" : "/login"}>
              <Button variant="secondary">
                <Settings className="h-4 w-4" strokeWidth={2} />
                {user ? "Open Settings" : "Sign in to configure"}
              </Button>
            </Link>
          </CardBody>
        </Card>
      )}

      {error ? (
        <Card className="border-bear/30 bg-bear/5">
          <CardBody className="flex flex-col items-center gap-3 py-12 text-center">
            <AlertTriangle className="h-7 w-7 text-bear" strokeWidth={1.75} />
            <p className="text-sm text-muted">{error}</p>
            <Button variant="secondary" onClick={load}>
              <RefreshCw className="h-4 w-4" strokeWidth={2} />
              Retry
            </Button>
          </CardBody>
        </Card>
      ) : loading && !data ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Card key={i}>
              <CardBody>
                <Skeleton className="h-64 w-full" />
              </CardBody>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {data?.commodities.map((c) => (
            <CommodityCard key={c.key} c={c} />
          ))}
        </div>
      )}

      <p className="mt-5 text-2xs leading-relaxed text-faint">
        Rule-based technical signals fused with LLM-scored news sentiment. Confidence is a model conviction score, not a
        guarantee. For research, not investment advice.
      </p>
    </div>
  );
}
