"use client";

import { useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { ChevronDown, ChevronUp, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, GradePill, DirectionTag, ScoreMeter, Skeleton } from "@/components/ui/primitives";
import { scoreColor } from "@/lib/grades";
import { fmtNumber, fmtPct, fmtPrice } from "@/lib/format";
import { QuickAddTrade } from "@/components/portfolio/QuickAddTrade";
import type { ScanRow } from "@/lib/engine/types";

type SortKey =
  | "ticker"
  | "setupFamily"
  | "pattern"
  | "grade"
  | "score"
  | "direction"
  | "currentPrice"
  | "distancePct"
  | "rsScore"
  | "rsiState"
  | "volumeRatio"
  | "rrRatio";

type SortDir = "asc" | "desc";

const GRADE_RANK: Record<string, number> = { "A+": 5, A: 4, B: 3, C: 2, Reject: 1 };

interface Column {
  key: SortKey;
  label: string;
  numeric: boolean;
}

const COLUMNS: Column[] = [
  { key: "ticker", label: "Ticker", numeric: false },
  { key: "setupFamily", label: "Setup", numeric: false },
  { key: "pattern", label: "Pattern", numeric: false },
  { key: "grade", label: "Grade", numeric: false },
  { key: "score", label: "Score", numeric: true },
  { key: "direction", label: "Dir", numeric: false },
  { key: "currentPrice", label: "Price", numeric: true },
  { key: "distancePct", label: "Dist%", numeric: true },
  { key: "rsScore", label: "RS", numeric: true },
  { key: "rsiState", label: "RSI", numeric: false },
  { key: "volumeRatio", label: "VolX", numeric: true },
  { key: "rrRatio", label: "R:R", numeric: true },
];

const rowKey = (row: ScanRow) => `${row.ticker}|${row.setupFamily}`;

function compareValues(a: ScanRow, b: ScanRow, key: SortKey): number {
  switch (key) {
    case "ticker":
      return a.ticker.localeCompare(b.ticker);
    case "setupFamily":
      return a.setupFamily.localeCompare(b.setupFamily);
    case "pattern":
      return a.pattern.localeCompare(b.pattern);
    case "direction":
      return a.direction.localeCompare(b.direction);
    case "rsiState":
      return (a.rsiState ?? "").localeCompare(b.rsiState ?? "");
    case "grade":
      return (GRADE_RANK[a.grade] ?? 0) - (GRADE_RANK[b.grade] ?? 0);
    case "score":
      return a.score - b.score;
    case "currentPrice":
      return a.currentPrice - b.currentPrice;
    case "distancePct":
      return (a.distancePct ?? -Infinity) - (b.distancePct ?? -Infinity);
    case "rsScore":
      return (a.rsScore ?? -Infinity) - (b.rsScore ?? -Infinity);
    case "volumeRatio":
      return (a.volumeRatio ?? -Infinity) - (b.volumeRatio ?? -Infinity);
    case "rrRatio":
      return (a.rrRatio ?? -Infinity) - (b.rrRatio ?? -Infinity);
    default:
      return 0;
  }
}

function rsiTone(state: string | null): "bull" | "bear" | "warn" | "neutral" {
  if (!state) return "neutral";
  const s = state.toLowerCase();
  if (s === "overbought") return "warn";
  if (s === "oversold") return "bear";
  if (s === "strong" || s === "healthy") return "bull";
  return "neutral";
}

export function ResultsTable({
  rows,
  selectedKey,
  onSelect,
  loading,
}: {
  rows: ScanRow[];
  selectedKey: string | null;
  onSelect: (row: ScanRow) => void;
  loading: boolean;
}) {
  const reduce = useReducedMotion();
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const cmp = compareValues(a, b, sortKey);
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "ticker" || key === "pattern" || key === "setupFamily" ? "asc" : "desc");
    }
  };

  const ariaSort = (key: SortKey): "ascending" | "descending" | "none" =>
    key === sortKey ? (sortDir === "asc" ? "ascending" : "descending") : "none";

  const headerCell = (col: Column) => {
    const SortIcon =
      col.key === sortKey ? (sortDir === "asc" ? ChevronUp : ChevronDown) : ChevronsUpDown;
    return (
      <th
        key={col.key}
        scope="col"
        aria-sort={ariaSort(col.key)}
        className={cn(
          "whitespace-nowrap px-3 py-2.5 text-2xs font-semibold uppercase tracking-wider text-faint",
          col.numeric ? "text-right" : "text-left",
        )}
      >
        <button
          type="button"
          onClick={() => toggleSort(col.key)}
          className={cn(
            "inline-flex items-center gap-1 rounded transition-colors hover:text-text",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
            col.numeric && "flex-row-reverse",
            col.key === sortKey && "text-text",
          )}
        >
          <span>{col.label}</span>
          <SortIcon className="h-3 w-3" strokeWidth={2} aria-hidden />
        </button>
      </th>
    );
  };

  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-surface shadow-card">
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 z-10 bg-elevated">
          <tr className="border-b border-border">
            {COLUMNS.map(headerCell)}
            <th scope="col" className="whitespace-nowrap px-3 py-2.5 text-right text-2xs font-semibold uppercase tracking-wider text-faint">
              Trade
            </th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            Array.from({ length: 8 }).map((_, i) => (
              <tr key={i} className="border-b border-border/60">
                {COLUMNS.map((col) => (
                  <td key={col.key} className="px-3 py-3">
                    <Skeleton className={cn("h-4", col.numeric ? "ml-auto w-12" : "w-20")} />
                  </td>
                ))}
                <td className="px-3 py-3"><Skeleton className="ml-auto h-4 w-16" /></td>
              </tr>
            ))
          ) : sorted.length === 0 ? (
            <tr>
              <td colSpan={COLUMNS.length + 1} className="px-4 py-16 text-center">
                <p className="text-sm text-muted">
                  No setups cleared your filters — lower the minimum score or widen the universe.
                </p>
              </td>
            </tr>
          ) : (
            sorted.map((row, i) => {
              const key = rowKey(row);
              const selected = key === selectedKey;
              return (
                <motion.tr
                  key={key}
                  initial={reduce ? false : { opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: reduce ? 0 : 0.2, delay: reduce ? 0 : Math.min(i * 0.03, 0.4), ease: "easeOut" }}
                  role="button"
                  tabIndex={0}
                  aria-pressed={selected}
                  aria-label={`${row.ticker} ${row.setupFamily} setup, grade ${row.grade}, score ${row.score.toFixed(0)}`}
                  onClick={() => onSelect(row)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(row);
                    }
                  }}
                  className={cn(
                    "cursor-pointer border-b border-border/60 transition-colors odd:bg-[rgb(var(--zebra)/0.03)] hover:bg-[rgb(var(--hover)/0.06)]",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand/60",
                    selected && "bg-brand/10 ring-1 ring-inset ring-brand/40",
                  )}
                >
                  {/* Ticker + market badge */}
                  <td className="whitespace-nowrap px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-bold text-text tnum">{row.ticker}</span>
                      <Badge tone="neutral">{row.market}</Badge>
                    </div>
                  </td>
                  {/* Setup family */}
                  <td className="whitespace-nowrap px-3 py-2.5 capitalize text-muted">{row.setupFamily}</td>
                  {/* Pattern */}
                  <td className="whitespace-nowrap px-3 py-2.5 text-muted">
                    <div>{row.pattern || "—"}</div>
                    {row.chartPattern && (
                      <div className="text-2xs font-medium text-brand/90">{row.chartPattern}</div>
                    )}
                  </td>
                  {/* Grade */}
                  <td className="whitespace-nowrap px-3 py-2.5">
                    <GradePill grade={row.grade} />
                  </td>
                  {/* Score meter */}
                  <td className="px-3 py-2.5">
                    <ScoreMeter score={row.score} className="min-w-[7rem]" />
                  </td>
                  {/* Direction */}
                  <td className="whitespace-nowrap px-3 py-2.5">
                    <DirectionTag direction={row.direction} />
                  </td>
                  {/* Price */}
                  <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-text tnum">
                    {fmtPrice(row.currentPrice, row.market)}
                  </td>
                  {/* Dist% */}
                  <td
                    className={cn(
                      "whitespace-nowrap px-3 py-2.5 text-right font-mono tnum",
                      row.distancePct == null
                        ? "text-faint"
                        : row.distancePct >= 0
                          ? "text-bull"
                          : "text-bear",
                    )}
                  >
                    {fmtPct(row.distancePct)}
                  </td>
                  {/* RS */}
                  <td className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", scoreColor(row.rsScore ?? 0))}>
                    {row.rsScore == null ? "—" : fmtNumber(row.rsScore, 0)}
                  </td>
                  {/* RSI state */}
                  <td className="whitespace-nowrap px-3 py-2.5">
                    {row.rsiState ? (
                      <Badge tone={rsiTone(row.rsiState)} className="capitalize">
                        {row.rsiState}
                      </Badge>
                    ) : (
                      <span className="text-faint">—</span>
                    )}
                  </td>
                  {/* VolX */}
                  <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-muted tnum">
                    {row.volumeRatio == null ? "—" : `${fmtNumber(row.volumeRatio, 1)}×`}
                  </td>
                  {/* R:R */}
                  <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-muted tnum">
                    {row.rrRatio == null ? "—" : `${fmtNumber(row.rrRatio, 1)}R`}
                  </td>
                  {/* Quick add to paper trade */}
                  <td className="whitespace-nowrap px-3 py-2.5 text-right">
                    <QuickAddTrade
                      ticker={row.ticker}
                      market={row.market}
                      entry={row.entry}
                      stop={row.stop}
                      target={row.target2r}
                      setupFamily={row.setupFamily}
                      score={row.score}
                      grade={row.grade}
                      pattern={row.pattern}
                      source="scan"
                    />
                  </td>
                </motion.tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
