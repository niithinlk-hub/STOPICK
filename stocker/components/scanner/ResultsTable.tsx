"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { ChevronDown, ChevronUp, ChevronsUpDown, SlidersHorizontal, Check } from "lucide-react";
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
  /** Locked columns (identity columns) always show and can't be toggled off. */
  locked?: boolean;
  /** Renders the row's <td> for this column. */
  render: (row: ScanRow) => ReactNode;
}

function rsiTone(state: string | null): "bull" | "bear" | "warn" | "neutral" {
  if (!state) return "neutral";
  const s = state.toLowerCase();
  if (s === "overbought") return "warn";
  if (s === "oversold") return "bear";
  if (s === "strong" || s === "healthy") return "bull";
  return "neutral";
}

const COLUMNS: Column[] = [
  {
    key: "ticker",
    label: "Ticker",
    numeric: false,
    locked: true,
    render: (row) => (
      <td key="ticker" className="whitespace-nowrap px-3 py-2.5">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-bold text-text tnum">{row.ticker}</span>
          <Badge tone="neutral">{row.market}</Badge>
        </div>
      </td>
    ),
  },
  {
    key: "setupFamily",
    label: "Setup",
    numeric: false,
    locked: true,
    render: (row) => (
      <td key="setupFamily" className="whitespace-nowrap px-3 py-2.5 capitalize text-muted">
        {row.setupFamily}
      </td>
    ),
  },
  {
    key: "pattern",
    label: "Pattern",
    numeric: false,
    locked: true,
    render: (row) => (
      <td key="pattern" className="whitespace-nowrap px-3 py-2.5 text-muted">
        <div>{row.pattern || "—"}</div>
        {row.chartPattern && <div className="text-2xs font-medium text-brand/90">{row.chartPattern}</div>}
      </td>
    ),
  },
  {
    key: "grade",
    label: "Grade",
    numeric: false,
    locked: true,
    render: (row) => (
      <td key="grade" className="whitespace-nowrap px-3 py-2.5">
        <GradePill grade={row.grade} />
      </td>
    ),
  },
  {
    key: "score",
    label: "Score",
    numeric: true,
    render: (row) => (
      <td key="score" className="px-3 py-2.5">
        <ScoreMeter score={row.score} className="min-w-[7rem]" />
      </td>
    ),
  },
  {
    key: "direction",
    label: "Dir",
    numeric: false,
    render: (row) => (
      <td key="direction" className="whitespace-nowrap px-3 py-2.5">
        <DirectionTag direction={row.direction} />
      </td>
    ),
  },
  {
    key: "currentPrice",
    label: "Price",
    numeric: true,
    render: (row) => (
      <td key="currentPrice" className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-text tnum">
        {fmtPrice(row.currentPrice, row.market)}
      </td>
    ),
  },
  {
    key: "distancePct",
    label: "Dist%",
    numeric: true,
    render: (row) => (
      <td
        key="distancePct"
        className={cn(
          "whitespace-nowrap px-3 py-2.5 text-right font-mono tnum",
          row.distancePct == null ? "text-faint" : row.distancePct >= 0 ? "text-bull" : "text-bear",
        )}
      >
        {fmtPct(row.distancePct)}
      </td>
    ),
  },
  {
    key: "rsScore",
    label: "RS",
    numeric: true,
    render: (row) => (
      <td key="rsScore" className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", scoreColor(row.rsScore ?? 0))}>
        {row.rsScore == null ? "—" : fmtNumber(row.rsScore, 0)}
      </td>
    ),
  },
  {
    key: "rsiState",
    label: "RSI",
    numeric: false,
    render: (row) => (
      <td key="rsiState" className="whitespace-nowrap px-3 py-2.5">
        {row.rsiState ? (
          <Badge tone={rsiTone(row.rsiState)} className="capitalize">
            {row.rsiState}
          </Badge>
        ) : (
          <span className="text-faint">—</span>
        )}
      </td>
    ),
  },
  {
    key: "volumeRatio",
    label: "VolX",
    numeric: true,
    render: (row) => (
      <td key="volumeRatio" className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-muted tnum">
        {row.volumeRatio == null ? "—" : `${fmtNumber(row.volumeRatio, 1)}×`}
      </td>
    ),
  },
  {
    key: "rrRatio",
    label: "R:R",
    numeric: true,
    render: (row) => (
      <td key="rrRatio" className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-muted tnum">
        {row.rrRatio == null ? "—" : `${fmtNumber(row.rrRatio, 1)}R`}
      </td>
    ),
  },
];

const OPTIONAL_COLUMNS = COLUMNS.filter((c) => !c.locked);
const ALL_KEYS = COLUMNS.map((c) => c.key);
const STORAGE_KEY = "stocker-scan-columns";

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

  // Which optional columns are shown. Defaults to all; persisted per browser.
  const [visible, setVisible] = useState<Set<SortKey>>(() => new Set(ALL_KEYS));
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  // Load saved selection after mount (avoids SSR/client hydration mismatch).
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as string[];
      const next = new Set<SortKey>(COLUMNS.filter((c) => c.locked).map((c) => c.key));
      for (const k of saved) if (OPTIONAL_COLUMNS.some((c) => c.key === k)) next.add(k as SortKey);
      setVisible(next);
    } catch {
      /* ignore malformed storage */
    }
  }, []);

  // Close the column menu on outside click / Escape.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  const persist = (next: Set<SortKey>) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(OPTIONAL_COLUMNS.filter((c) => next.has(c.key)).map((c) => c.key)));
    } catch {
      /* ignore */
    }
  };

  const toggleColumn = (key: SortKey) => {
    const next = new Set(visible);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    // Keep at least one optional column visible.
    if (OPTIONAL_COLUMNS.every((c) => !next.has(c.key))) return;
    persist(next);
    setVisible(next);
    // If we just hid the active sort column, fall back to a visible one.
    if (key === sortKey && !next.has(key)) {
      const fallback = COLUMNS.find((c) => c.locked || next.has(c.key));
      if (fallback) setSortKey(fallback.key);
    }
  };

  const setAll = (on: boolean) => {
    const next = on
      ? new Set<SortKey>(ALL_KEYS)
      : new Set<SortKey>(COLUMNS.filter((c) => c.locked).map((c) => c.key).concat(OPTIONAL_COLUMNS[0].key));
    persist(next);
    setVisible(next);
  };

  const displayColumns = useMemo(() => COLUMNS.filter((c) => c.locked || visible.has(c.key)), [visible]);
  const hiddenCount = OPTIONAL_COLUMNS.length - OPTIONAL_COLUMNS.filter((c) => visible.has(c.key)).length;

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
    const SortIcon = col.key === sortKey ? (sortDir === "asc" ? ChevronUp : ChevronDown) : ChevronsUpDown;
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
    <div className="space-y-2">
      {/* Column chooser toolbar */}
      <div className="flex items-center justify-end">
        <div className="relative" ref={menuRef}>
          <button
            type="button"
            onClick={() => setMenuOpen((o) => !o)}
            aria-haspopup="true"
            aria-expanded={menuOpen}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-2xs font-semibold uppercase tracking-wider text-muted",
              "transition-colors hover:text-text hover:border-brand/50",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
            )}
          >
            <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
            Columns
            {hiddenCount > 0 && (
              <span className="rounded bg-brand/15 px-1 text-2xs font-bold text-brand">{hiddenCount} hidden</span>
            )}
          </button>
          {menuOpen && (
            <div
              role="menu"
              className="absolute right-0 z-20 mt-1 w-56 rounded-xl border border-border bg-elevated p-2 shadow-pop"
            >
              <div className="flex items-center justify-between px-1.5 pb-1.5">
                <span className="text-2xs font-semibold uppercase tracking-wider text-faint">Show columns</span>
                <div className="flex gap-2 text-2xs">
                  <button type="button" onClick={() => setAll(true)} className="text-brand hover:underline">
                    All
                  </button>
                  <button type="button" onClick={() => setAll(false)} className="text-muted hover:underline">
                    Min
                  </button>
                </div>
              </div>
              <p className="px-1.5 pb-1 text-2xs text-faint">Ticker · Setup · Pattern · Grade always shown.</p>
              {OPTIONAL_COLUMNS.map((col) => {
                const on = visible.has(col.key);
                return (
                  <button
                    key={col.key}
                    type="button"
                    role="menuitemcheckbox"
                    aria-checked={on}
                    onClick={() => toggleColumn(col.key)}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-lg px-1.5 py-1.5 text-left text-sm transition-colors hover:bg-[rgb(var(--hover)/0.08)]",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
                    )}
                  >
                    <span
                      className={cn(
                        "flex h-4 w-4 items-center justify-center rounded border",
                        on ? "border-brand bg-brand text-white" : "border-border text-transparent",
                      )}
                    >
                      <Check className="h-3 w-3" strokeWidth={3} aria-hidden />
                    </span>
                    <span className={cn(on ? "text-text" : "text-muted")}>{col.label}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-border bg-surface shadow-card">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-elevated">
            <tr className="border-b border-border">
              {displayColumns.map(headerCell)}
              <th scope="col" className="whitespace-nowrap px-3 py-2.5 text-right text-2xs font-semibold uppercase tracking-wider text-faint">
                Trade
              </th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i} className="border-b border-border/60">
                  {displayColumns.map((col) => (
                    <td key={col.key} className="px-3 py-3">
                      <Skeleton className={cn("h-4", col.numeric ? "ml-auto w-12" : "w-20")} />
                    </td>
                  ))}
                  <td className="px-3 py-3">
                    <Skeleton className="ml-auto h-4 w-16" />
                  </td>
                </tr>
              ))
            ) : sorted.length === 0 ? (
              <tr>
                <td colSpan={displayColumns.length + 1} className="px-4 py-16 text-center">
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
                    {displayColumns.map((col) => col.render(row))}
                    {/* Quick add to paper trade (always shown) */}
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
    </div>
  );
}
