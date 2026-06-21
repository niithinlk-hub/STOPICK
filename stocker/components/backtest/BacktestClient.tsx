"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, Play, RotateCcw, Search } from "lucide-react";
import { getBacktest, type BacktestApiResult } from "@/lib/client/api";
import type { BacktestTrade, WalkForwardRow } from "@/lib/engine/backtest";
import type { Market } from "@/lib/engine/types";
import { cn } from "@/lib/cn";
import { fmtNumber, fmtPct, fmtPrice } from "@/lib/format";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  Skeleton,
  Stat,
} from "@/components/ui/primitives";

/* Chart colors — hex that matches the design tokens (recharts needs literals). */
const C = {
  brand: "#22d3ee",
  bull: "#34d399",
  bear: "#f87171",
  warn: "#fbbf24",
  muted: "#94a3b8",
  border: "#26303e",
} as const;

const MARKETS: { value: Market; label: string }[] = [
  { value: "US", label: "US" },
  { value: "NSE", label: "NSE" },
];

type Status = "idle" | "loading" | "ready" | "error";

function fmtDate(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleDateString();
}

/* exitReason -> badge tone + label */
function exitTone(reason: string): { tone: "bull" | "bear" | "neutral"; label: string } {
  switch (reason) {
    case "target_2r":
      return { tone: "bull", label: "Target 2R" };
    case "stop":
      return { tone: "bear", label: "Stop" };
    case "time_stop":
      return { tone: "neutral", label: "Time stop" };
    default:
      return { tone: "neutral", label: reason };
  }
}

/* ------------------------------ Chart shell ------------------------------ */

function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number; color?: string }[];
  label?: string | number;
  formatter?: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-overlay/95 px-3 py-2 shadow-pop backdrop-blur">
      {label !== undefined && <div className="mb-1 text-2xs uppercase tracking-widest text-faint">{label}</div>}
      <div className="space-y-0.5">
        {payload.map((p, i) => (
          <div key={i} className="flex items-center gap-2 font-mono text-xs tnum">
            {p.color && <span className="h-2 w-2 rounded-sm" style={{ background: p.color }} />}
            {p.name && <span className="text-muted">{p.name}</span>}
            <span className="font-semibold text-text">
              {formatter && typeof p.value === "number" ? formatter(p.value) : p.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------ Main client ------------------------------ */

export function BacktestClient() {
  const reduce = useReducedMotion();
  const [symbol, setSymbol] = useState("");
  const [market, setMarket] = useState<Market>("US");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<BacktestApiResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ranSymbol, setRanSymbol] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const enter = reduce
    ? { initial: { opacity: 1 }, animate: { opacity: 1 } }
    : { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 } };

  const run = useCallback(
    async (sym: string, mkt: Market) => {
      const ticker = sym.trim().toUpperCase();
      if (!ticker) return;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setStatus("loading");
      setError(null);
      setRanSymbol(ticker);
      try {
        const data = await getBacktest(ticker, mkt, controller.signal);
        if (controller.signal.aborted) return;
        setResult(data);
        setStatus("ready");
      } catch (e) {
        if (controller.signal.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
        setError(e instanceof Error ? e.message : "Backtest failed.");
        setStatus("error");
      }
    },
    [],
  );

  const onSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      void run(symbol, market);
    },
    [run, symbol, market],
  );

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6">
      <header className="mb-5">
        <h1 className="text-lg font-semibold tracking-tight">Backtest</h1>
        <p className="mt-1 text-sm text-muted">
          Walk-forward breakout backtest with next-bar execution, ATR stops, a 2R target, and Monte
          Carlo trade-sequence resampling.
        </p>
      </header>

      {/* Controls */}
      <Card className="mb-5">
        <CardBody>
          <form onSubmit={onSubmit} className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1">
              <label htmlFor="bt-symbol" className="mb-1.5 block text-2xs font-medium uppercase tracking-widest text-faint">
                Ticker
              </label>
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" strokeWidth={2} />
                <input
                  id="bt-symbol"
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  placeholder={market === "NSE" ? "e.g. RELIANCE" : "e.g. AAPL"}
                  autoComplete="off"
                  spellCheck={false}
                  className={cn(
                    "w-full rounded-lg border border-border bg-elevated py-2 pl-9 pr-3 font-mono text-sm uppercase tracking-wide text-text tnum",
                    "placeholder:font-sans placeholder:normal-case placeholder:tracking-normal placeholder:text-faint",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
                  )}
                />
              </div>
            </div>

            <div>
              <span className="mb-1.5 block text-2xs font-medium uppercase tracking-widest text-faint">Market</span>
              <div className="inline-flex rounded-lg border border-border bg-elevated p-0.5" role="group" aria-label="Market">
                {MARKETS.map((m) => {
                  const active = market === m.value;
                  return (
                    <button
                      key={m.value}
                      type="button"
                      aria-pressed={active}
                      onClick={() => setMarket(m.value)}
                      className={cn(
                        "rounded-md px-3.5 py-1.5 font-mono text-xs font-semibold transition-colors tnum",
                        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
                        active ? "bg-brand/15 text-brand ring-1 ring-inset ring-brand/30" : "text-muted hover:text-text",
                      )}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <Button type="submit" disabled={status === "loading" || !symbol.trim()} className="sm:w-auto">
              <Play className="h-4 w-4" strokeWidth={2} />
              {status === "loading" ? "Running…" : "Run backtest"}
            </Button>
          </form>
        </CardBody>
      </Card>

      <div aria-live="polite">
        {status === "idle" && <EmptyState />}
        {status === "loading" && <LoadingState />}
        {status === "error" && <ErrorState message={error} symbol={ranSymbol} onRetry={() => void run(ranSymbol, market)} />}
        {status === "ready" && result && (
          <AnimatePresence mode="wait">
            {result.trades.length === 0 ? (
              <NoTradesState key="no-trades" symbol={ranSymbol} />
            ) : (
              <motion.div
                key={`${result.ticker}-${result.market}`}
                initial={enter.initial}
                animate={enter.animate}
                transition={{ duration: reduce ? 0 : 0.22, ease: "easeOut" }}
                className="space-y-5"
              >
                <Results result={result} reduce={!!reduce} />
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

/* -------------------------------- Results -------------------------------- */

function Results({ result, reduce }: { result: BacktestApiResult; reduce: boolean }) {
  const { summary, equityCurve, walkForward, monteCarlo, mcStats, trades, market } = result;

  return (
    <>
      <div className="flex items-baseline gap-2">
        <h2 className="font-mono text-base font-semibold tracking-tight text-text tnum">{result.ticker}</h2>
        <Badge tone="brand">{market}</Badge>
        <span className="text-xs text-faint">breakout, next-bar, 2R target</span>
      </div>

      {/* Summary stat cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        <Stat label="Trades" value={fmtNumber(summary.trades, 0)} />
        <Stat label="Win rate" value={`${fmtNumber(summary.winRate, 1)}%`} />
        <Stat label="Expectancy R" value={fmtNumber(summary.expectancyR, 2)} />
        <Stat label="Profit factor" value={fmtNumber(summary.profitFactor, 2)} />
        <Stat label="Max DD" value={`${fmtNumber(summary.maxDrawdownPct, 1)}%`} />
        <Stat label="Sharpe-like" value={fmtNumber(summary.sharpeLike, 2)} />
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <EquityCurveCard data={equityCurve} />
        <WalkForwardCard rows={walkForward} />
      </div>

      <MonteCarloCard data={monteCarlo} stats={mcStats} />

      <TradesCard trades={trades} market={market} reduce={reduce} />
    </>
  );
}

/* ----------------------------- Equity curve ------------------------------ */

function EquityCurveCard({ data }: { data: { index: number; equity: number }[] }) {
  const chartData = useMemo(() => data.map((d) => ({ trade: d.index + 1, equity: d.equity })), [data]);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Equity curve</CardTitle>
        <span className="text-2xs uppercase tracking-widest text-faint">growth of 1.0</span>
      </CardHeader>
      <CardBody>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
              <defs>
                <linearGradient id="bt-equity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.brand} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={C.brand} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={C.border} strokeOpacity={0.4} vertical={false} />
              <XAxis
                dataKey="trade"
                tick={{ fill: C.muted, fontSize: 11 }}
                stroke={C.border}
                tickLine={false}
                axisLine={{ stroke: C.border }}
              />
              <YAxis
                tick={{ fill: C.muted, fontSize: 11 }}
                stroke={C.border}
                tickLine={false}
                axisLine={false}
                width={44}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip
                cursor={{ stroke: C.border }}
                content={<ChartTooltip formatter={(v) => v.toFixed(3)} />}
              />
              <Area
                type="monotone"
                dataKey="equity"
                name="Equity"
                stroke={C.brand}
                strokeWidth={2}
                fill="url(#bt-equity)"
                dot={false}
                activeDot={{ r: 3, fill: C.brand }}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardBody>
    </Card>
  );
}

/* ------------------------------ Walk-forward ----------------------------- */

const SEGMENT_LABEL: Record<string, string> = {
  in_sample: "In-sample",
  out_of_sample: "Out-of-sample",
};

function WalkForwardCard({ rows }: { rows: WalkForwardRow[] }) {
  const data = useMemo(
    () =>
      rows.map((r) => ({
        segment: SEGMENT_LABEL[r.segment] ?? r.segment,
        winRate: r.winRate,
        expectancyR: r.expectancyR,
      })),
    [rows],
  );
  return (
    <Card>
      <CardHeader>
        <CardTitle>Walk-forward split</CardTitle>
        <span className="text-2xs uppercase tracking-widest text-faint">in vs out-of-sample</span>
      </CardHeader>
      <CardBody>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }} barGap={6}>
              <CartesianGrid stroke={C.border} strokeOpacity={0.4} vertical={false} />
              <XAxis
                dataKey="segment"
                tick={{ fill: C.muted, fontSize: 11 }}
                stroke={C.border}
                tickLine={false}
                axisLine={{ stroke: C.border }}
              />
              <YAxis
                tick={{ fill: C.muted, fontSize: 11 }}
                stroke={C.border}
                tickLine={false}
                axisLine={false}
                width={44}
              />
              <Tooltip cursor={{ fill: "rgba(148,163,184,0.08)" }} content={<ChartTooltip formatter={(v) => v.toFixed(2)} />} />
              <Bar dataKey="winRate" name="Win rate %" fill={C.brand} radius={[3, 3, 0, 0]} isAnimationActive={false} />
              <Bar dataKey="expectancyR" name="Expectancy R" fill={C.warn} radius={[3, 3, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex items-center justify-center gap-4 text-2xs text-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ background: C.brand }} /> Win rate %
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ background: C.warn }} /> Expectancy R
          </span>
        </div>
      </CardBody>
    </Card>
  );
}

/* ------------------------------ Monte Carlo ------------------------------ */

function MonteCarloCard({
  data,
  stats,
}: {
  data: { run: number; endingR: number; worstR: number }[];
  stats: { p5: number; p50: number; p95: number };
}) {
  const buckets = useMemo(() => {
    if (!data.length) return [] as { bucket: number; label: string; count: number }[];
    const values = data.map((d) => d.endingR);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binCount = 24;
    const span = max - min || 1;
    const width = span / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => {
      const center = min + width * (i + 0.5);
      return { bucket: Number(center.toFixed(2)), label: center.toFixed(1), count: 0 };
    });
    for (const v of values) {
      let idx = Math.floor((v - min) / width);
      if (idx >= binCount) idx = binCount - 1;
      if (idx < 0) idx = 0;
      bins[idx].count += 1;
    }
    return bins;
  }, [data]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Monte Carlo distribution</CardTitle>
        <div className="flex items-center gap-3 font-mono text-2xs text-muted tnum">
          <span>
            <span className="text-faint">p5</span> {stats.p5.toFixed(2)}R
          </span>
          <span>
            <span className="text-faint">p50</span> {stats.p50.toFixed(2)}R
          </span>
          <span>
            <span className="text-faint">p95</span> {stats.p95.toFixed(2)}R
          </span>
        </div>
      </CardHeader>
      <CardBody>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={buckets} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
              <CartesianGrid stroke={C.border} strokeOpacity={0.4} vertical={false} />
              <XAxis
                dataKey="bucket"
                type="number"
                domain={["dataMin", "dataMax"]}
                tick={{ fill: C.muted, fontSize: 11 }}
                stroke={C.border}
                tickLine={false}
                axisLine={{ stroke: C.border }}
                tickFormatter={(v: number) => `${v.toFixed(1)}R`}
              />
              <YAxis
                tick={{ fill: C.muted, fontSize: 11 }}
                stroke={C.border}
                tickLine={false}
                axisLine={false}
                width={36}
                allowDecimals={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(148,163,184,0.08)" }}
                content={<ChartTooltip formatter={(v) => `${v} runs`} />}
              />
              <ReferenceLine x={stats.p5} stroke={C.bear} strokeDasharray="4 3" label={{ value: "p5", fill: C.bear, fontSize: 10, position: "top" }} />
              <ReferenceLine x={stats.p50} stroke={C.muted} strokeDasharray="4 3" label={{ value: "p50", fill: C.muted, fontSize: 10, position: "top" }} />
              <ReferenceLine x={stats.p95} stroke={C.bull} strokeDasharray="4 3" label={{ value: "p95", fill: C.bull, fontSize: 10, position: "top" }} />
              <Bar dataKey="count" name="Runs" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {buckets.map((b, i) => (
                  <Cell key={i} fill={b.bucket >= 0 ? C.bull : C.bear} fillOpacity={0.75} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-2xs text-faint">
          200 resampled trade sequences. Each bar counts how many runs ended in that cumulative-R bucket.
        </p>
      </CardBody>
    </Card>
  );
}

/* -------------------------------- Trades --------------------------------- */

function TradesCard({ trades, market, reduce }: { trades: BacktestTrade[]; market: Market; reduce: boolean }) {
  const rows = trades;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Trades</CardTitle>
        <span className="font-mono text-2xs text-faint tnum">{fmtNumber(rows.length, 0)} trades</span>
      </CardHeader>
      <CardBody className="p-0">
        <div className="max-h-[28rem] overflow-auto">
          <table className="w-full border-collapse text-sm">
            <thead className="sticky top-0 z-10 bg-elevated">
              <tr className="text-left text-2xs uppercase tracking-widest text-faint">
                <th scope="col" className="px-4 py-2.5 font-medium">Entry</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Exit</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Entry px</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Exit px</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Return</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">R</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Bars</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Exit reason</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((t, i) => {
                const tone = exitTone(t.exitReason);
                const tr = (
                  <>
                    <td className="px-4 py-2 font-mono text-xs text-muted tnum">{fmtDate(t.entryTime)}</td>
                    <td className="px-4 py-2 font-mono text-xs text-muted tnum">{fmtDate(t.exitTime)}</td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-text tnum">{fmtPrice(t.entryPrice, market)}</td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-text tnum">{fmtPrice(t.exitPrice, market)}</td>
                    <td className={cn("px-4 py-2 text-right font-mono text-xs font-semibold tnum", t.returnPct >= 0 ? "text-bull" : "text-bear")}>
                      {fmtPct(t.returnPct)}
                    </td>
                    <td className={cn("px-4 py-2 text-right font-mono text-xs font-semibold tnum", t.rMultiple >= 0 ? "text-bull" : "text-bear")}>
                      {fmtNumber(t.rMultiple, 2)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-muted tnum">{fmtNumber(t.barsHeld, 0)}</td>
                    <td className="px-4 py-2">
                      <Badge tone={tone.tone}>{tone.label}</Badge>
                    </td>
                  </>
                );
                return reduce ? (
                  <tr key={i} className="border-t border-border/60 odd:bg-white/[0.015] hover:bg-white/[0.03]">
                    {tr}
                  </tr>
                ) : (
                  <motion.tr
                    key={i}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.2, ease: "easeOut", delay: Math.min(i, 12) * 0.03 }}
                    className="border-t border-border/60 odd:bg-white/[0.015] hover:bg-white/[0.03]"
                  >
                    {tr}
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardBody>
    </Card>
  );
}

/* ---------------------------- State components --------------------------- */

function EmptyState() {
  return (
    <Card>
      <CardBody className="flex flex-col items-center justify-center gap-3 py-16 text-center">
        <div className="grid h-12 w-12 place-items-center rounded-full bg-brand/10 ring-1 ring-inset ring-brand/20">
          <Search className="h-5 w-5 text-brand" strokeWidth={2} />
        </div>
        <p className="text-sm font-medium text-text">Enter a ticker to run a walk-forward breakout backtest</p>
        <p className="max-w-md text-xs text-muted">
          The engine simulates next-bar entries on breakouts, ATR-based stops, a 2R target, then
          resamples trades via Monte Carlo to gauge outcome dispersion.
        </p>
      </CardBody>
    </Card>
  );
}

function NoTradesState({ symbol }: { symbol: string }) {
  return (
    <Card>
      <CardBody className="flex flex-col items-center justify-center gap-3 py-16 text-center">
        <div className="grid h-12 w-12 place-items-center rounded-full bg-warn/10 ring-1 ring-inset ring-warn/20">
          <AlertTriangle className="h-5 w-5 text-warn" strokeWidth={2} />
        </div>
        <p className="text-sm font-medium text-text">
          No qualifying breakouts for <span className="font-mono tnum">{symbol}</span>
        </p>
        <p className="max-w-md text-xs text-muted">
          The strategy produced zero trades over the available history. Try another ticker or market.
        </p>
      </CardBody>
    </Card>
  );
}

function ErrorState({ message, symbol, onRetry }: { message: string | null; symbol: string; onRetry: () => void }) {
  return (
    <Card>
      <CardBody className="flex flex-col items-center justify-center gap-3 py-16 text-center">
        <div className="grid h-12 w-12 place-items-center rounded-full bg-bear/10 ring-1 ring-inset ring-bear/20">
          <AlertTriangle className="h-5 w-5 text-bear" strokeWidth={2} />
        </div>
        <p className="text-sm font-medium text-text">
          Backtest failed{symbol ? <> for <span className="font-mono tnum">{symbol}</span></> : null}
        </p>
        <p className="max-w-md text-xs text-muted">{message ?? "Something went wrong while running the backtest."}</p>
        <Button variant="secondary" onClick={onRetry}>
          <RotateCcw className="h-4 w-4" strokeWidth={2} />
          Retry
        </Button>
      </CardBody>
    </Card>
  );
}

function LoadingState() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <Skeleton className="h-80" />
        <Skeleton className="h-80" />
      </div>
      <Skeleton className="h-80" />
      <Skeleton className="h-64" />
    </div>
  );
}
