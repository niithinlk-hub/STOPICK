"use client";

import { useCallback, useEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, Gauge, RefreshCw } from "lucide-react";
import { scanStocks } from "@/lib/client/api";
import { cn } from "@/lib/cn";
import type { Market, RegimeSignal } from "@/lib/engine/types";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  DirectionTag,
  ScoreGauge,
  ScoreMeter,
  Skeleton,
} from "@/components/ui/primitives";

/* ----------------------------- helpers ----------------------------- */

const MARKET_ORDER: Market[] = ["US", "NSE"];

const MARKET_LABELS: Record<Market, string> = {
  US: "United States",
  NSE: "India (NSE)",
};

type VolTone = "info" | "neutral" | "warn";

function volStyle(state: string): { tone: VolTone; label: string } {
  switch (state) {
    case "compressed":
      return { tone: "info", label: "Compressed" };
    case "expanding":
      return { tone: "warn", label: "Expanding" };
    case "normal":
      return { tone: "neutral", label: "Normal" };
    default:
      return { tone: "neutral", label: "Unknown" };
  }
}

/** Direction → ring + glow color class for the big indicator. */
function directionRing(direction: RegimeSignal["direction"]): string {
  switch (direction) {
    case "bullish":
      return "ring-bull/40 bg-bull/10";
    case "bearish":
      return "ring-bear/40 bg-bear/10";
    default:
      return "ring-border bg-elevated";
  }
}

/* ---------------------------- regime card --------------------------- */

function RegimeCard({ regime, index }: { regime: RegimeSignal; index: number }) {
  const reduce = useReducedMotion();
  const vol = volStyle(regime.volatilityState);

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0 : 0.24, ease: "easeOut", delay: reduce ? 0 : index * 0.06 }}
    >
      <Card>
        <CardHeader>
          <div className="flex flex-col">
            <CardTitle>{MARKET_LABELS[regime.market]}</CardTitle>
            <span className="mt-0.5 text-2xs uppercase tracking-widest text-faint">
              Benchmark{" "}
              <span className="font-mono tnum text-muted">{regime.benchmarkSymbol}</span>
            </span>
          </div>
          <Badge tone={vol.tone}>{vol.label} vol</Badge>
        </CardHeader>

        <CardBody className="space-y-6">
          {/* Direction + trend gauge */}
          <div className="flex items-center justify-between gap-4">
            <div
              className={cn(
                "flex flex-col items-center justify-center gap-2 rounded-2xl px-6 py-5 ring-1 ring-inset",
                directionRing(regime.direction),
              )}
            >
              <span className="text-2xs uppercase tracking-widest text-faint">Direction</span>
              <DirectionTag direction={regime.direction} className="text-sm" />
            </div>

            <div className="flex flex-col items-center gap-1">
              <ScoreGauge score={regime.trendStrength} size={104} />
              <span className="text-2xs uppercase tracking-widest text-faint">Trend strength</span>
            </div>
          </div>

          {/* Breadth proxy meter */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-2xs font-medium uppercase tracking-widest text-faint">
                Breadth proxy
              </span>
            </div>
            <ScoreMeter score={regime.breadthLikeProxy} />
          </div>

          {/* Explanation */}
          <p className="rounded-lg border border-border bg-elevated/50 p-3 text-xs leading-relaxed text-muted">
            {regime.explanation}
          </p>
        </CardBody>
      </Card>
    </motion.div>
  );
}

/* ------------------------------ legend ------------------------------ */

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-2xs text-faint">
      <span className="font-medium uppercase tracking-widest">Volatility states</span>
      <span className="flex items-center gap-1.5">
        <Badge tone="info">Compressed</Badge>
        <span className="text-muted">coiled / low range</span>
      </span>
      <span className="flex items-center gap-1.5">
        <Badge tone="neutral">Normal</Badge>
        <span className="text-muted">steady range</span>
      </span>
      <span className="flex items-center gap-1.5">
        <Badge tone="warn">Expanding</Badge>
        <span className="text-muted">elevated risk</span>
      </span>
    </div>
  );
}

/* --------------------------- loading state -------------------------- */

function LoadingState() {
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      {[0, 1].map((i) => (
        <Card key={i}>
          <CardHeader>
            <div className="flex flex-col gap-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-24" />
            </div>
            <Skeleton className="h-5 w-20" />
          </CardHeader>
          <CardBody className="space-y-6">
            <div className="flex items-center justify-between gap-4">
              <Skeleton className="h-24 w-32 rounded-2xl" />
              <Skeleton className="h-[104px] w-[104px] rounded-full" />
            </div>
            <Skeleton className="h-2 w-full rounded-full" />
            <Skeleton className="h-16 w-full" />
          </CardBody>
        </Card>
      ))}
    </div>
  );
}

/* ------------------------------ client ------------------------------ */

export function RegimeClient() {
  const [regimes, setRegimes] = useState<Record<string, RegimeSignal> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const res = await scanStocks(
        {
          country: "BOTH",
          source: "sample",
          timeframe: "1d",
          setupMode: "both",
          minScore: 100,
        },
        signal,
      );
      if (signal?.aborted) return;
      setRegimes(res.regimes);
    } catch (err) {
      if (signal?.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
      setError(err instanceof Error ? err.message : "Failed to load market regime.");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    void load(ctrl.signal);
    return () => ctrl.abort();
  }, [load]);

  const ordered = regimes
    ? MARKET_ORDER.map((m) => regimes[m]).filter((r): r is RegimeSignal => Boolean(r))
    : [];

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-brand/10 ring-1 ring-inset ring-brand/30">
            <Gauge className="h-5 w-5 text-brand" strokeWidth={2} />
          </span>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Market Regime</h1>
            <p className="mt-0.5 text-sm text-muted">
              Top-down read on each market — direction, trend strength, volatility state and a
              breadth-like proxy that frame every setup.
            </p>
          </div>
        </div>
        <Button
          variant="secondary"
          onClick={() => void load()}
          disabled={loading}
          aria-label="Refresh market regime"
        >
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} strokeWidth={2} />
          Refresh
        </Button>
      </div>

      <div className="mt-5" aria-live="polite">
        {loading && <LoadingState />}

        {!loading && error && (
          <Card>
            <CardBody className="flex flex-col items-center gap-4 py-12 text-center">
              <span className="grid h-12 w-12 place-items-center rounded-full bg-bear/10 ring-1 ring-inset ring-bear/30">
                <AlertTriangle className="h-6 w-6 text-bear" strokeWidth={2} />
              </span>
              <div>
                <div className="text-sm font-semibold text-text">Could not load regime data</div>
                <p className="mt-1 max-w-sm text-xs text-muted">{error}</p>
              </div>
              <Button variant="secondary" onClick={() => void load()}>
                <RefreshCw className="h-4 w-4" strokeWidth={2} />
                Retry
              </Button>
            </CardBody>
          </Card>
        )}

        {!loading && !error && ordered.length === 0 && (
          <Card>
            <CardBody className="flex flex-col items-center gap-3 py-12 text-center">
              <span className="grid h-12 w-12 place-items-center rounded-full bg-elevated ring-1 ring-inset ring-border">
                <Gauge className="h-6 w-6 text-faint" strokeWidth={2} />
              </span>
              <div className="text-sm font-semibold text-text">No regime data available</div>
              <p className="max-w-sm text-xs text-muted">
                The scan returned no market regime signals. Try refreshing in a moment.
              </p>
            </CardBody>
          </Card>
        )}

        {!loading && !error && ordered.length > 0 && (
          <div className="space-y-5">
            <div className="grid gap-5 lg:grid-cols-2">
              {ordered.map((regime, i) => (
                <RegimeCard key={regime.market} regime={regime} index={i} />
              ))}
            </div>
            <Legend />
          </div>
        )}
      </div>
    </div>
  );
}
