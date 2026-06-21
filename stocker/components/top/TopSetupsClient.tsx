"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  AlertTriangle,
  ArrowUpRight,
  Crosshair,
  Gauge as GaugeIcon,
  Layers,
  RefreshCw,
  Target,
  TrendingUp,
  Volume2,
  X,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { GRADE_ORDER } from "@/lib/config";
import { fmtPrice, fmtNumber } from "@/lib/format";
import { scanStocks } from "@/lib/client/api";
import type { Country, Grade, ScanParams, ScanRow } from "@/lib/engine/types";
import {
  Badge,
  Button,
  Card,
  DirectionTag,
  GradePill,
  ScoreGauge,
  Skeleton,
} from "@/components/ui/primitives";

/* ------------------------------- controls ------------------------------- */

type UniverseKey = "sample" | "tier_1" | "tier_2";
type GradeFilter = "all" | "a-up" | "aplus";

const UNIVERSES: { key: UniverseKey; label: string; hint: string }[] = [
  { key: "sample", label: "Sample 15", hint: "Curated quick scan" },
  { key: "tier_1", label: "Set 1 · 500", hint: "Top 500 of market" },
  { key: "tier_2", label: "Set 2 · 500", hint: "Next 500 of market" },
];

const COUNTRIES: { key: Country; label: string }[] = [
  { key: "US", label: "US" },
  { key: "NSE", label: "NSE" },
  { key: "BOTH", label: "Both" },
];

const GRADE_FILTERS: { key: GradeFilter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "a-up", label: "A & up" },
  { key: "aplus", label: "A+ only" },
];

const GRADE_RANK: Record<Grade, number> = GRADE_ORDER.reduce(
  (acc, g, i) => ({ ...acc, [g]: i }),
  {} as Record<Grade, number>,
);

function passesGradeFilter(grade: Grade, filter: GradeFilter): boolean {
  if (filter === "aplus") return grade === "A+";
  if (filter === "a-up") return grade === "A+" || grade === "A";
  return true;
}

/* ----------------------------- segmented UI ----------------------------- */

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { key: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-2xs font-medium uppercase tracking-widest text-faint">{label}</span>
      <div
        role="group"
        aria-label={label}
        className="inline-flex rounded-lg border border-border bg-elevated/60 p-0.5"
      >
        {options.map((o) => {
          const active = o.key === value;
          return (
            <button
              key={o.key}
              type="button"
              aria-pressed={active}
              onClick={() => onChange(o.key)}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
                active ? "bg-brand/15 text-text ring-1 ring-inset ring-brand/30" : "text-muted hover:text-text",
              )}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* -------------------------------- chips --------------------------------- */

function Chip({ icon: Icon, label, value, tone = "neutral" }: {
  icon: LucideIcon;
  label: string;
  value: string;
  tone?: "neutral" | "bull" | "bear" | "warn" | "info";
}) {
  const toneText: Record<string, string> = {
    neutral: "text-text",
    bull: "text-bull",
    bear: "text-bear",
    warn: "text-warn",
    info: "text-info",
  };
  return (
    <div className="flex items-center justify-between rounded-md border border-border bg-elevated/40 px-2 py-1.5">
      <span className="inline-flex items-center gap-1 text-2xs font-medium uppercase tracking-wide text-faint">
        <Icon className="h-3 w-3" strokeWidth={2} />
        {label}
      </span>
      <span className={cn("font-mono text-xs font-semibold tnum", toneText[tone])}>{value}</span>
    </div>
  );
}

/* --------------------------- card sub-helpers --------------------------- */

function rsTone(rs: number | null): "neutral" | "bull" | "bear" {
  if (rs === null) return "neutral";
  if (rs >= 60) return "bull";
  if (rs < 40) return "bear";
  return "neutral";
}

function rsiTone(state: string | null): "neutral" | "bull" | "warn" {
  if (!state) return "neutral";
  if (state === "overbought") return "warn";
  if (state === "strong" || state === "healthy") return "bull";
  return "neutral";
}

function marketTone(market: ScanRow["market"]): "info" | "warn" {
  return market === "US" ? "info" : "warn";
}

function SetupCard({ row, index, reduce }: { row: ScanRow; index: number; reduce: boolean }) {
  const emphasised = row.grade === "A+" || row.grade === "A";
  const rr = row.rrRatio !== null ? `${fmtNumber(row.rrRatio, 1)}R` : "—";
  const volx = row.volumeRatio !== null ? `${fmtNumber(row.volumeRatio, 1)}x` : "—";
  const rs = row.rsScore !== null ? fmtNumber(row.rsScore, 0) : "—";
  const rsi = row.rsiState ? row.rsiState.charAt(0).toUpperCase() + row.rsiState.slice(1) : "—";

  return (
    <motion.div
      layout={!reduce}
      initial={reduce ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={reduce ? { opacity: 0 } : { opacity: 0, y: -8 }}
      transition={{ duration: reduce ? 0 : 0.25, ease: "easeOut", delay: reduce ? 0 : Math.min(index, 12) * 0.03 }}
    >
      <Link
        href="/"
        aria-label={`Open ${row.ticker} in the scanner`}
        className={cn(
          "group block h-full rounded-xl outline-none",
          "focus-visible:ring-2 focus-visible:ring-brand/60",
        )}
      >
        <Card
          className={cn(
            "flex h-full flex-col gap-3 p-4 transition-colors group-hover:border-border-strong",
            emphasised && "ring-1 ring-inset ring-bull/25",
          )}
        >
          {/* header */}
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate font-mono text-base font-bold tracking-tight text-text tnum">
                  {row.ticker}
                </span>
                <Badge tone={marketTone(row.market)}>{row.market}</Badge>
              </div>
              <div className="mt-1 flex items-center gap-2">
                <GradePill grade={row.grade} />
                <DirectionTag direction={row.direction} />
              </div>
            </div>
            <ScoreGauge score={row.score} size={64} />
          </div>

          {/* pattern + sector */}
          <div className="flex items-center justify-between gap-2 border-y border-border py-2">
            <span className="inline-flex items-center gap-1.5 truncate text-xs font-medium text-text">
              <Layers className="h-3.5 w-3.5 text-brand" strokeWidth={2} />
              {row.pattern}
            </span>
            <span className="shrink-0 text-2xs uppercase tracking-wide text-faint">
              {row.setupFamily} · {row.timeframe}
            </span>
          </div>

          {/* key chips */}
          <div className="grid grid-cols-2 gap-2">
            <Chip icon={TrendingUp} label="RS" value={rs} tone={rsTone(row.rsScore)} />
            <Chip icon={GaugeIcon} label="RSI" value={rsi} tone={rsiTone(row.rsiState)} />
            <Chip icon={Volume2} label="VolX" value={volx} tone={row.volumeRatio !== null && row.volumeRatio >= 1.5 ? "bull" : "neutral"} />
            <Chip icon={Target} label="R:R" value={rr} tone={row.rrRatio !== null && row.rrRatio >= 2 ? "bull" : "neutral"} />
          </div>

          {/* execution mini-row */}
          <div className="grid grid-cols-3 gap-2 rounded-lg bg-elevated/40 p-2">
            <ExecCell label="Entry" value={fmtPrice(row.entry, row.market)} tone="text-text" />
            <ExecCell label="Stop" value={fmtPrice(row.stop, row.market)} tone="text-bear" />
            <ExecCell label="Target 2R" value={fmtPrice(row.target2r, row.market)} tone="text-bull" />
          </div>

          {/* first reason */}
          {row.whyQualified && (
            <p className="flex items-start gap-1.5 text-2xs leading-relaxed text-muted">
              <Crosshair className="mt-0.5 h-3 w-3 shrink-0 text-brand" strokeWidth={2} />
              <span className="line-clamp-2">{row.whyQualified}</span>
            </p>
          )}
        </Card>
      </Link>
    </motion.div>
  );
}

function ExecCell({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[0.625rem] uppercase tracking-wide text-faint">{label}</span>
      <span className={cn("font-mono text-xs font-semibold tnum", tone)}>{value}</span>
    </div>
  );
}

/* --------------------------- loading skeleton --------------------------- */

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3" aria-hidden>
      {Array.from({ length: 6 }).map((_, i) => (
        <Card key={i} className="flex flex-col gap-3 p-4">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <Skeleton className="h-5 w-24" />
              <Skeleton className="h-4 w-28" />
            </div>
            <Skeleton className="h-16 w-16 rounded-full" />
          </div>
          <Skeleton className="h-8 w-full" />
          <div className="grid grid-cols-2 gap-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
          <Skeleton className="h-12 w-full" />
        </Card>
      ))}
    </div>
  );
}

/* -------------------------------- client -------------------------------- */

export function TopSetupsClient() {
  const reduce = useReducedMotion() ?? false;

  const [universe, setUniverse] = useState<UniverseKey>("sample");
  const [country, setCountry] = useState<Country>("US");
  const [gradeFilter, setGradeFilter] = useState<GradeFilter>("all");

  const [rows, setRows] = useState<ScanRow[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ qualified: number; scanned: number; elapsedMs: number } | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const runScan = useCallback(
    async (u: UniverseKey, c: Country) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setLoading(true);
      setError(null);

      const params: ScanParams = {
        country: c,
        source: u,
        timeframe: "1d",
        setupMode: "both",
        minScore: 0,
      };

      try {
        const res = await scanStocks(params, controller.signal);
        if (controller.signal.aborted) return;
        setRows(res.rows);
        setMeta({ qualified: res.qualifiedSymbols, scanned: res.scannedSymbols, elapsedMs: res.elapsedMs });
        setLoading(false);
      } catch (err) {
        if (controller.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
        setError(err instanceof Error ? err.message : "Scan failed. Please try again.");
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    runScan(universe, country);
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [universe, country]);

  const handleAbort = useCallback(() => {
    abortRef.current?.abort();
    setLoading(false);
    setError("Scan aborted.");
  }, []);

  const ranked = useMemo(() => {
    if (!rows) return [];
    return [...rows]
      .filter((r) => passesGradeFilter(r.grade, gradeFilter))
      .sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return GRADE_RANK[a.grade] - GRADE_RANK[b.grade];
      });
  }, [rows, gradeFilter]);

  const aTotal = useMemo(
    () => (rows ? rows.filter((r) => r.grade === "A+" || r.grade === "A").length : 0),
    [rows],
  );

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6">
      {/* header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-text">Top Setups</h1>
          <p className="mt-1 text-sm text-muted">
            The highest-conviction breakout and pullback candidates, ranked by composite score.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="secondary"
            onClick={() => runScan(universe, country)}
            disabled={loading}
            aria-label="Re-run scan"
          >
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} strokeWidth={2} />
            {loading ? "Scanning" : "Refresh"}
          </Button>
          {loading && (
            <Button variant="ghost" onClick={handleAbort} aria-label="Abort scan">
              <X className="h-4 w-4" strokeWidth={2} />
              Abort
            </Button>
          )}
        </div>
      </div>

      {/* controls */}
      <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-border bg-surface/60 px-4 py-3">
        <Segmented label="Universe" value={universe} options={UNIVERSES} onChange={setUniverse} />
        <Segmented label="Market" value={country} options={COUNTRIES} onChange={setCountry} />
        <Segmented label="Min grade" value={gradeFilter} options={GRADE_FILTERS} onChange={setGradeFilter} />
        <div className="ml-auto" aria-live="polite">
          {meta && !loading && !error && (
            <span className="font-mono text-2xs tnum text-faint">
              {ranked.length} shown · {aTotal} A-grade · {meta.qualified}/{meta.scanned} qualified ·{" "}
              {(meta.elapsedMs / 1000).toFixed(1)}s
            </span>
          )}
          {loading && <span className="text-2xs text-faint">Scanning universe…</span>}
        </div>
      </div>

      {/* body */}
      <div className="mt-5" aria-live="polite" aria-busy={loading}>
        {loading ? (
          <SkeletonGrid />
        ) : error ? (
          <Card className="flex flex-col items-center gap-3 p-10 text-center">
            <AlertTriangle className="h-8 w-8 text-warn" strokeWidth={1.75} />
            <div>
              <p className="text-sm font-medium text-text">Could not load setups</p>
              <p className="mt-1 text-xs text-muted">{error}</p>
            </div>
            <Button onClick={() => runScan(universe, country)}>
              <RefreshCw className="h-4 w-4" strokeWidth={2} />
              Retry
            </Button>
          </Card>
        ) : ranked.length === 0 ? (
          <Card className="flex flex-col items-center gap-3 p-10 text-center">
            <TrendingUp className="h-8 w-8 text-faint" strokeWidth={1.75} />
            <div>
              <p className="text-sm font-medium text-text">No setups cleared your filters</p>
              <p className="mt-1 text-xs text-muted">
                Try a wider universe, switch markets, or relax the minimum grade.
              </p>
            </div>
            {gradeFilter !== "all" && (
              <Button variant="secondary" onClick={() => setGradeFilter("all")}>
                Show all grades
              </Button>
            )}
          </Card>
        ) : (
          <motion.div layout={!reduce} className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            <AnimatePresence mode="popLayout">
              {ranked.map((row, i) => (
                <SetupCard key={`${row.ticker}-${row.market}-${row.setupFamily}`} row={row} index={i} reduce={reduce} />
              ))}
            </AnimatePresence>
          </motion.div>
        )}
      </div>

      {/* footnote */}
      <p className="mt-6 inline-flex items-center gap-1 text-2xs text-faint">
        <ArrowUpRight className="h-3 w-3" strokeWidth={2} />
        Select any card to open it in the Scanner for the full breakdown and chart.
      </p>
    </div>
  );
}
