"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, Crown, LineChart, Loader2, RefreshCw, Trophy } from "lucide-react";
import { cn } from "@/lib/cn";
import { fmtNumber } from "@/lib/format";
import { scoreColor } from "@/lib/grades";
import { scanStocks } from "@/lib/client/api";
import type {
  Country,
  ScanParams,
  ScanResponse,
  ScanRow,
  UniverseSource,
} from "@/lib/engine/types";
import {
  Badge,
  Button,
  Card,
  CardBody,
  DirectionTag,
  Skeleton,
  Stat,
} from "@/components/ui/primitives";

/* ------------------------------- controls ------------------------------- */

const COUNTRIES: { value: Country; label: string }[] = [
  { value: "BOTH", label: "Both" },
  { value: "NSE", label: "India" },
  { value: "US", label: "US" },
];

const UNIVERSES: { value: UniverseSource; label: string }[] = [
  { value: "tier_1", label: "Set 1" },
  { value: "tier_2", label: "Set 2" },
  { value: "sample", label: "Sample" },
];

type Status = "idle" | "loading" | "ready" | "error";

function buildParams(country: Country, source: UniverseSource): ScanParams {
  return {
    country,
    source,
    timeframe: "1d",
    setupMode: "both",
    minScore: 0,
    limit: 250,
  };
}

/* --------------------------- segmented control -------------------------- */

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-2xs font-medium uppercase tracking-widest text-faint">{label}</span>
      <div
        role="radiogroup"
        aria-label={label}
        className="inline-flex rounded-lg border border-border bg-elevated/60 p-0.5"
      >
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(opt.value)}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
                "disabled:cursor-not-allowed disabled:opacity-60",
                active ? "bg-brand/15 text-text ring-1 ring-inset ring-brand/30" : "text-muted hover:text-text",
              )}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------ RS helpers ------------------------------ */

const RSI_TONE: Record<string, "bull" | "info" | "warn" | "neutral"> = {
  overbought: "warn",
  strong: "bull",
  healthy: "info",
  oversold: "neutral",
};

function rsiTone(state: string | null): "bull" | "info" | "warn" | "neutral" {
  if (!state) return "neutral";
  return RSI_TONE[state.toLowerCase()] ?? "neutral";
}

function rsiLabel(state: string | null): string {
  if (!state) return "—";
  return state.charAt(0).toUpperCase() + state.slice(1);
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const MEDALS = ["text-warn", "text-faint", "text-bear"];

/* ------------------------------- main view ------------------------------ */

export function RelativeStrengthClient() {
  const reduceMotion = useReducedMotion();
  const [country, setCountry] = useState<Country>("BOTH");
  const [source, setSource] = useState<UniverseSource>("tier_1");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ScanResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const runScan = useCallback(
    async (nextCountry: Country, nextSource: UniverseSource) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setStatus("loading");
      setError(null);
      try {
        const res = await scanStocks(buildParams(nextCountry, nextSource), controller.signal);
        if (controller.signal.aborted) return;
        setData(res);
        setStatus("ready");
      } catch (err) {
        if (controller.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
        setError(err instanceof Error ? err.message : "Scan failed unexpectedly.");
        setStatus("error");
      }
    },
    [],
  );

  // Initial scan + re-scan on control change.
  useEffect(() => {
    runScan(country, source);
    return () => abortRef.current?.abort();
  }, [country, source, runScan]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setStatus((s) => (s === "loading" ? "idle" : s));
  }, []);

  const leaderboard = useMemo(() => {
    if (!data) return [];
    return data.rows
      .filter((r) => r.rsScore !== null && Number.isFinite(r.rsScore))
      .sort((a, b) => (b.rsScore ?? 0) - (a.rsScore ?? 0));
  }, [data]);

  const summary = useMemo(() => {
    const scores = leaderboard.map((r) => r.rsScore as number);
    const med = median(scores);
    const strongest = leaderboard[0] ?? null;
    return { count: leaderboard.length, median: med, strongest };
  }, [leaderboard]);

  const benchmarkLabel = country === "NSE" ? "^NSEI" : country === "US" ? "SPY" : "^NSEI / SPY";
  const loading = status === "loading";

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="space-y-1">
          <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
            <LineChart className="h-5 w-5 text-brand" strokeWidth={2} />
            Relative Strength
          </h1>
          <p className="text-sm text-muted">
            Names leading the market, ranked by composite RS score versus the broad benchmark.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <Segmented label="Market" value={country} options={COUNTRIES} onChange={setCountry} disabled={loading} />
          <Segmented label="Universe" value={source} options={UNIVERSES} onChange={setSource} disabled={loading} />
          {loading ? (
            <Button variant="secondary" onClick={cancel} aria-label="Cancel scan">
              <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} />
              Cancel
            </Button>
          ) : (
            <Button variant="secondary" onClick={() => runScan(country, source)} aria-label="Rescan relative strength">
              <RefreshCw className="h-4 w-4" strokeWidth={2} />
              Rescan
            </Button>
          )}
        </div>
      </div>

      {/* Async status (a11y) */}
      <p className="sr-only" aria-live="polite">
        {loading
          ? "Scanning the universe for relative strength leaders."
          : status === "ready"
            ? `${leaderboard.length} ranked names ready.`
            : status === "error"
              ? `Scan failed: ${error ?? "unknown error"}.`
              : ""}
      </p>

      {/* Summary stats */}
      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
        {loading && !data ? (
          <>
            <Skeleton className="h-[104px]" />
            <Skeleton className="h-[104px]" />
            <Skeleton className="h-[104px]" />
          </>
        ) : (
          <>
            <Stat label="Ranked Names" value={summary.count > 0 ? summary.count : "—"} hint="RS measured" />
            <Stat
              label="Median RS"
              value={summary.median !== null ? fmtNumber(summary.median, 1) : "—"}
              hint="0–100 composite"
            />
            <Stat
              label="Strongest"
              value={
                summary.strongest ? (
                  <span className="flex items-baseline gap-2">
                    <span className="truncate">{summary.strongest.ticker}</span>
                    <span className={cn("text-base", scoreColor(summary.strongest.rsScore ?? 0))}>
                      {fmtNumber(summary.strongest.rsScore, 1)}
                    </span>
                  </span>
                ) : (
                  "—"
                )
              }
              hint="top of the board"
            />
          </>
        )}
      </div>

      {/* Leaderboard */}
      <Card className="mt-5 overflow-hidden">
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <Trophy className="h-4 w-4 text-warn" strokeWidth={2} />
            <h2 className="text-sm font-semibold tracking-tight text-text">RS Leaderboard</h2>
          </div>
          {data && !loading && (
            <span className="font-mono text-2xs text-faint tnum">
              {data.successfulSymbols}/{data.scannedSymbols} scanned · {(data.elapsedMs / 1000).toFixed(1)}s
            </span>
          )}
        </div>

        <CardBody className="p-0">
          {status === "error" ? (
            <ErrorState message={error} onRetry={() => runScan(country, source)} />
          ) : loading && !data ? (
            <LoadingRows />
          ) : leaderboard.length === 0 ? (
            <EmptyState />
          ) : (
            <ul className="divide-y divide-border">
              <AnimatePresence initial={false}>
                {leaderboard.map((row, idx) => (
                  <LeaderRow key={`${row.ticker}-${row.market}`} row={row} rank={idx + 1} reduceMotion={!!reduceMotion} />
                ))}
              </AnimatePresence>
            </ul>
          )}
        </CardBody>
      </Card>

      {/* Methodology note */}
      <p className="mt-3 text-2xs leading-relaxed text-faint">
        RS is measured versus the broad benchmark ({benchmarkLabel}) using 1W / 1M / 3M alpha blended with
        path smoothness — a steady, persistent outperformer scores higher than a choppy one with the same
        net return.
      </p>
    </div>
  );
}

/* ------------------------------- leader row ----------------------------- */

function LeaderRow({ row, rank, reduceMotion }: { row: ScanRow; rank: number; reduceMotion: boolean }) {
  const rs = Math.max(0, Math.min(100, row.rsScore ?? 0));
  const isMedal = rank <= 3;
  const medalColor = isMedal ? MEDALS[rank - 1] : "text-faint";

  return (
    <motion.li
      layout={!reduceMotion}
      initial={reduceMotion ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -6 }}
      transition={{ duration: reduceMotion ? 0 : 0.22, ease: "easeOut", delay: reduceMotion ? 0 : Math.min(rank - 1, 12) * 0.03 }}
      className={cn(
        "grid grid-cols-[2.25rem_minmax(0,1fr)] items-center gap-x-4 gap-y-2 px-4 py-3 transition-colors hover:bg-white/[0.03]",
        "sm:grid-cols-[2.25rem_minmax(8rem,11rem)_minmax(0,1fr)_auto]",
        isMedal && "bg-warn/[0.03]",
      )}
    >
      {/* Rank */}
      <div className="flex items-center justify-center">
        {isMedal ? (
          <Crown className={cn("h-5 w-5", medalColor)} strokeWidth={2} aria-hidden />
        ) : null}
        <span
          className={cn(
            "font-mono text-sm font-semibold tnum",
            isMedal ? "sr-only" : "text-faint",
          )}
          aria-label={`Rank ${rank}`}
        >
          {rank}
        </span>
      </div>

      {/* Ticker + market */}
      <div className="flex min-w-0 items-center gap-2">
        <span className="truncate font-mono text-sm font-semibold tnum text-text">{row.ticker}</span>
        <Badge tone="neutral">{row.market}</Badge>
      </div>

      {/* RS strength bar */}
      <div className="col-span-2 flex items-center gap-3 sm:col-span-1">
        <div className="h-2 w-full overflow-hidden rounded-full bg-elevated" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(rs)} aria-label={`Relative strength ${rs.toFixed(0)} of 100`}>
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${rs}%`, background: "linear-gradient(90deg, rgb(var(--brand-2)), rgb(var(--brand)))" }}
          />
        </div>
        <span className={cn("w-10 shrink-0 text-right font-mono text-sm font-semibold tnum", scoreColor(rs))}>
          {fmtNumber(row.rsScore, 1)}
        </span>
      </div>

      {/* Tags + setup score */}
      <div className="col-span-2 flex flex-wrap items-center justify-start gap-x-4 gap-y-1.5 sm:col-span-1 sm:justify-end">
        <Badge tone={rsiTone(row.rsiState)}>RSI {rsiLabel(row.rsiState)}</Badge>
        <DirectionTag direction={row.direction} />
        <span className="inline-flex items-baseline gap-1">
          <span className="text-2xs uppercase tracking-wide text-faint">score</span>
          <span className={cn("font-mono text-sm font-semibold tnum", scoreColor(row.score))}>{row.score.toFixed(0)}</span>
        </span>
      </div>
    </motion.li>
  );
}

/* ---------------------------- supporting states ------------------------- */

function LoadingRows() {
  return (
    <ul className="divide-y divide-border" aria-hidden>
      {Array.from({ length: 10 }).map((_, i) => (
        <li key={i} className="grid grid-cols-[2.25rem_minmax(8rem,11rem)_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3.5">
          <Skeleton className="h-4 w-4" />
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-2.5 w-full" />
          <Skeleton className="h-4 w-28" />
        </li>
      ))}
    </ul>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-16 text-center">
      <span className="grid h-11 w-11 place-items-center rounded-xl bg-elevated">
        <LineChart className="h-5 w-5 text-faint" strokeWidth={2} />
      </span>
      <p className="text-sm font-medium text-text">No relative-strength reads available</p>
      <p className="max-w-sm text-xs text-muted">
        None of the scanned symbols returned a computable RS score for this universe. Try a broader universe
        or a different market.
      </p>
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string | null; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-4 py-16 text-center">
      <span className="grid h-11 w-11 place-items-center rounded-xl bg-bear/10 ring-1 ring-inset ring-bear/30">
        <AlertTriangle className="h-5 w-5 text-bear" strokeWidth={2} />
      </span>
      <div className="space-y-1">
        <p className="text-sm font-medium text-text">Could not load the leaderboard</p>
        <p className="max-w-sm text-xs text-muted">{message ?? "The scan failed unexpectedly."}</p>
      </div>
      <Button variant="secondary" onClick={onRetry} aria-label="Retry scan">
        <RefreshCw className="h-4 w-4" strokeWidth={2} />
        Retry
      </Button>
    </div>
  );
}
