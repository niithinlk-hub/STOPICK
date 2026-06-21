"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, ChevronDown, Info, Radar, RefreshCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { scanStocks } from "@/lib/client/api";
import { GRADE_ORDER } from "@/lib/config";
import { fmtNumber } from "@/lib/format";
import {
  Button,
  Card,
  CardBody,
  GradePill,
  Stat,
} from "@/components/ui/primitives";
import { ScanControls } from "@/components/scanner/ScanControls";
import { ResultsTable } from "@/components/scanner/ResultsTable";
import { SetupDetail } from "@/components/scanner/SetupDetail";
import type { Country, Grade, ScanParams, ScanResponse, ScanRow, SetupSignal } from "@/lib/engine/types";

const DEFAULT_PARAMS: ScanParams = {
  country: "US",
  source: "sample",
  timeframe: "1d",
  setupMode: "both",
  minScore: 75,
};

const rowKey = (row: Pick<ScanRow, "ticker" | "setupFamily">) => `${row.ticker}|${row.setupFamily}`;

function topGrade(rows: ScanRow[]): Grade | null {
  for (const g of GRADE_ORDER) {
    if (rows.some((r) => r.grade === g)) return g;
  }
  return null;
}

function avgScore(rows: ScanRow[]): number {
  if (rows.length === 0) return 0;
  return rows.reduce((sum, r) => sum + r.score, 0) / rows.length;
}

export function ScannerClient() {
  const reduce = useReducedMotion();
  const [params, setParams] = useState<ScanParams>(DEFAULT_PARAMS);
  const [response, setResponse] = useState<ScanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [showFailures, setShowFailures] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const searchParams = useSearchParams();
  const autoRan = useRef(false);

  const runScanWith = useCallback(async (p: ScanParams) => {
    // Cancel any in-flight scan.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setSelectedKey(null);
    setShowFailures(false);

    try {
      const data = await scanStocks(p, controller.signal);
      if (controller.signal.aborted) return;
      setResponse(data);
    } catch (err) {
      if (controller.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) {
        return;
      }
      setError(err instanceof Error ? err.message : "Scan failed. Please try again.");
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        abortRef.current = null;
      }
    }
  }, []);

  const runScan = useCallback(() => runScanWith(params), [runScanWith, params]);

  // Deep-link from the watchlist: /?symbols=A,B&country=US prefills a manual scan
  // and runs it once on arrival.
  useEffect(() => {
    if (autoRan.current) return;
    const symbols = searchParams.get("symbols");
    if (!symbols) return;
    autoRan.current = true;
    const c = searchParams.get("country");
    const country: Country = c === "NSE" ? "NSE" : c === "BOTH" ? "BOTH" : "US";
    const next: ScanParams = { ...DEFAULT_PARAMS, country, source: "manual", manualSymbols: symbols, minScore: 0 };
    setParams(next);
    runScanWith(next);
  }, [searchParams, runScanWith]);

  const selectedSetup: SetupSignal | null = useMemo(() => {
    if (!response || !selectedKey) return null;
    return (
      response.setups.find((s) => rowKey(s) === selectedKey) ?? null
    );
  }, [response, selectedKey]);

  const handleSelect = useCallback((row: ScanRow) => {
    setSelectedKey((prev) => (prev === rowKey(row) ? null : rowKey(row)));
  }, []);

  const clearSelection = useCallback(() => setSelectedKey(null), []);

  const rows = response?.rows ?? [];
  const failureEntries = response ? Object.entries(response.failures) : [];
  const hasResults = response !== null;
  const showEmptyHero = !hasResults && !loading && !error;

  const best = topGrade(rows);

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6">
      {/* Header */}
      <header className="mb-5">
        <h1 className="text-lg font-semibold tracking-tight">Scanner</h1>
        <p className="mt-0.5 text-sm text-muted">
          Scan a universe for high-confidence breakout and pullback setups, ranked by composite score.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[18rem_minmax(0,1fr)]">
        {/* Left controls rail */}
        <div className="lg:sticky lg:top-5 lg:self-start">
          <ScanControls params={params} onChange={setParams} onRun={runScan} loading={loading} />
        </div>

        {/* Main column */}
        <div className="min-w-0 space-y-5">
          {/* aria-live status */}
          <div className="sr-only" role="status" aria-live="polite">
            {loading
              ? "Scanning the universe for setups…"
              : error
                ? `Scan failed: ${error}`
                : response
                  ? `Scan complete. ${response.qualifiedSymbols} setups qualified from ${response.scannedSymbols} symbols.`
                  : ""}
          </div>

          {/* Summary stat cards */}
          {(hasResults || loading) && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat
                label="Scanned"
                value={loading ? "…" : fmtNumber(response?.scannedSymbols ?? 0, 0)}
                hint={response ? `${response.successfulSymbols} ok` : undefined}
              />
              <Stat
                label="Qualified"
                value={loading ? "…" : fmtNumber(response?.qualifiedSymbols ?? rows.length, 0)}
              />
              <Stat label="Avg Score" value={loading ? "…" : fmtNumber(avgScore(rows), 1)} />
              <Stat
                label="Top Grade"
                value={loading ? "…" : best ? <GradePill grade={best} /> : "—"}
              />
            </div>
          )}

          {/* Notes */}
          {response && response.notes.length > 0 && (
            <Card className="border-info/30 bg-info/5">
              <CardBody className="flex gap-2.5 py-3">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-info" strokeWidth={2} aria-hidden />
                <ul className="space-y-1 text-xs leading-relaxed text-muted">
                  {response.notes.map((note, i) => (
                    <li key={i}>{note}</li>
                  ))}
                </ul>
              </CardBody>
            </Card>
          )}

          {/* Error state */}
          {error && !loading && (
            <Card className="border-bear/30 bg-bear/5">
              <CardBody className="flex flex-col items-start gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex gap-2.5">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-bear" strokeWidth={2} aria-hidden />
                  <div>
                    <div className="text-sm font-medium text-text">Scan failed</div>
                    <div className="text-xs text-muted">{error}</div>
                  </div>
                </div>
                <Button type="button" variant="secondary" onClick={runScan}>
                  <RefreshCw className="h-4 w-4" strokeWidth={2} aria-hidden />
                  Retry
                </Button>
              </CardBody>
            </Card>
          )}

          {/* Empty hero — invite first scan */}
          {showEmptyHero ? (
            <Card>
              <CardBody className="flex flex-col items-center gap-4 py-16 text-center">
                <span className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 ring-1 ring-brand/20">
                  <Radar className="h-7 w-7 text-brand" strokeWidth={1.75} aria-hidden />
                </span>
                <div className="max-w-sm space-y-1">
                  <h2 className="text-base font-semibold text-text">Ready to scan</h2>
                  <p className="text-sm text-muted">
                    Pick your universe and filters on the left, then run a scan to surface qualifying
                    setups ranked by score.
                  </p>
                </div>
                <Button type="button" onClick={runScan}>
                  <Radar className="h-4 w-4" strokeWidth={2} aria-hidden />
                  Run scan
                </Button>
              </CardBody>
            </Card>
          ) : (
            (hasResults || loading) && (
              <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_22rem]">
                {/* Results table */}
                <div className="min-w-0 space-y-3">
                  <ResultsTable
                    rows={rows}
                    selectedKey={selectedKey}
                    onSelect={handleSelect}
                    loading={loading}
                  />

                  {/* Failures (collapsible) */}
                  {failureEntries.length > 0 && (
                    <Card>
                      <button
                        type="button"
                        onClick={() => setShowFailures((v) => !v)}
                        aria-expanded={showFailures}
                        className={cn(
                          "flex w-full items-center justify-between gap-2 px-4 py-3 text-left transition-colors hover:bg-white/[0.02]",
                          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand/60",
                        )}
                      >
                        <span className="flex items-center gap-2 text-xs font-medium text-muted">
                          <AlertTriangle className="h-3.5 w-3.5 text-warn" strokeWidth={2} aria-hidden />
                          {failureEntries.length} symbol{failureEntries.length === 1 ? "" : "s"} failed to scan
                        </span>
                        <ChevronDown
                          className={cn("h-4 w-4 text-faint transition-transform", showFailures && "rotate-180")}
                          strokeWidth={2}
                          aria-hidden
                        />
                      </button>
                      {showFailures && (
                        <div className="max-h-56 overflow-y-auto border-t border-border px-4 py-3">
                          <ul className="space-y-1.5 text-2xs">
                            {failureEntries.map(([symbol, reason]) => (
                              <li key={symbol} className="flex gap-2">
                                <span className="font-mono font-semibold text-muted tnum">{symbol}</span>
                                <span className="text-faint">{reason}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </Card>
                  )}
                </div>

                {/* Detail column */}
                <div className="min-w-0 xl:sticky xl:top-5 xl:self-start">
                  {selectedSetup ? (
                    <motion.div
                      key={selectedKey}
                      initial={reduce ? false : { opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: reduce ? 0 : 0.22, ease: "easeOut" }}
                    >
                      <SetupDetail setup={selectedSetup} onClose={clearSelection} />
                    </motion.div>
                  ) : (
                    <Card>
                      <CardBody className="flex flex-col items-center gap-2 py-12 text-center">
                        <Radar className="h-6 w-6 text-faint" strokeWidth={1.75} aria-hidden />
                        <p className="text-sm text-muted">Select a setup to inspect</p>
                        <p className="max-w-[14rem] text-xs text-faint">
                          Click any row to see its score breakdown, reasons and execution plan.
                        </p>
                      </CardBody>
                    </Card>
                  )}
                </div>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
