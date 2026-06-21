"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Check, ExternalLink, RefreshCw, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { fmtNumber, fmtPct, fmtPrice } from "@/lib/format";
import { getSymbolDetail } from "@/lib/client/api";
import { tradingViewUrl } from "@/lib/tradingview";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  GradePill,
  ScoreGauge,
  Skeleton,
} from "@/components/ui/primitives";
import { CandleChart } from "@/components/scanner/CandleChart";
import { ConfidenceBreakdown } from "@/components/scanner/ConfidenceBreakdown";
import { ExecutionPlanCard } from "@/components/scanner/ExecutionPlanCard";
import { SetupActions } from "@/components/portfolio/SetupActions";
import type { Bar, SetupSignal } from "@/lib/engine/types";

/**
 * Full setup detail panel. The passed-in `setup` is the source of truth for all
 * scores / plan / reasons; the fetch is only used to pull chart bars (and refresh
 * the latest setup snapshot when present). On error the chart shows a retry.
 */
export function SetupDetail({ setup, onClose }: { setup: SetupSignal; onClose?: () => void }) {
  const [bars, setBars] = useState<Bar[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    getSymbolDetail(setup.ticker, setup.market, setup.timeframe, setup.setupFamily, controller.signal)
      .then((res) => {
        if (controller.signal.aborted) return;
        setBars(res.bars ?? []);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : "Failed to load chart data.");
        setLoading(false);
      });

    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setup.ticker, setup.market, setup.timeframe, setup.setupFamily, attempt]);

  const momentum = setup.momentum;
  const liquidity = setup.liquidity;

  const keyStats: { label: string; value: React.ReactNode }[] = [
    { label: "Pattern", value: setup.breakout?.patternName || setup.structure?.structureType || "—" },
    {
      label: "Chart Pattern",
      value: setup.chartPattern
        ? `${setup.chartPattern.name} · ${fmtNumber(setup.chartPattern.confidence, 0)}`
        : "—",
    },
    {
      label: "Trend",
      value: (
        <span className="capitalize">
          {setup.trend.direction} · {fmtNumber(setup.trend.strengthScore, 0)}
        </span>
      ),
    },
    { label: "RS Score", value: setup.relativeStrength ? fmtNumber(setup.relativeStrength.score, 0) : "—" },
    {
      label: "RSI",
      value: momentum ? (
        <span>
          {fmtNumber(momentum.rsi, 0)} <span className="text-faint">{momentum.rsiState}</span>
        </span>
      ) : (
        "—"
      ),
    },
    { label: "MACD Hist", value: momentum ? fmtNumber(momentum.macdHistogram, 3) : "—" },
    {
      label: "Liquidity",
      value: liquidity ? (
        <Badge tone={liquidity.tradable ? "bull" : "warn"}>{liquidity.tradable ? "Tradable" : "Thin"}</Badge>
      ) : (
        "—"
      ),
    },
    { label: "ATR %ile", value: fmtNumber(setup.atrPercentile, 0) },
    { label: "Event Risk", value: setup.eventRiskDays !== null ? `${setup.eventRiskDays}d` : "—" },
    { label: "VolX", value: setup.volume ? `${fmtNumber(setup.volume.volumeRatio, 2)}×` : "—" },
  ];

  return (
    <div className="flex max-h-[calc(100vh-2rem)] flex-col gap-4 overflow-y-auto pr-1 lg:max-h-[calc(100vh-6rem)]">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="font-mono text-lg font-bold tnum text-text">{setup.ticker}</span>
          <Badge tone="neutral">{setup.market}</Badge>
          <GradePill grade={setup.grade} />
          <span className="text-xs capitalize text-muted">{setup.setupFamily} · {setup.timeframe}</span>
        </div>
        <div className="flex items-center gap-3">
          <ScoreGauge score={setup.score} size={72} />
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="rounded-lg p-1.5 text-muted transition-colors hover:bg-white/[0.05] hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
            >
              <X className="h-5 w-5" strokeWidth={2} />
            </button>
          )}
        </div>
      </div>

      {/* (a) Chart */}
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center gap-2">
            <CardTitle>Price Action</CardTitle>
            {setup.chartPattern && (
              <Badge
                tone={
                  setup.chartPattern.direction === "bullish"
                    ? "bull"
                    : setup.chartPattern.direction === "bearish"
                      ? "bear"
                      : "neutral"
                }
              >
                {setup.chartPattern.name} · {fmtNumber(setup.chartPattern.confidence, 0)}
              </Badge>
            )}
          </div>
          <a
            href={tradingViewUrl(setup.ticker, setup.market, setup.exchange)}
            target="_blank"
            rel="noopener noreferrer"
            title={`Open ${setup.ticker} on TradingView`}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-border bg-elevated/60 px-2.5 py-1 text-xs font-medium text-text transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
          >
            TradingView
            <ExternalLink className="h-3.5 w-3.5 text-brand" strokeWidth={2} />
          </a>
        </CardHeader>
        <CardBody>
          {loading ? (
            <Skeleton className="h-[320px] w-full" />
          ) : error ? (
            <div className="flex h-[320px] flex-col items-center justify-center gap-3 text-center" aria-live="polite">
              <AlertTriangle className="h-7 w-7 text-warn" strokeWidth={1.75} />
              <p className="text-sm text-muted">{error}</p>
              <Button variant="secondary" onClick={() => setAttempt((a) => a + 1)}>
                <RefreshCw className="h-4 w-4" strokeWidth={2} />
                Retry
              </Button>
            </div>
          ) : (
            <CandleChart bars={bars} setup={setup} />
          )}
        </CardBody>
      </Card>

      {/* (b) Why this qualified */}
      <Card>
        <CardHeader>
          <CardTitle>Why this qualified</CardTitle>
        </CardHeader>
        <CardBody className="space-y-3">
          <ul className="space-y-1.5">
            {setup.reasonsFor.length > 0 ? (
              setup.reasonsFor.map((r, i) => (
                <li key={`for-${i}`} className="flex items-start gap-2 text-xs text-text">
                  <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-bull" strokeWidth={2} />
                  <span>{r}</span>
                </li>
              ))
            ) : (
              <li className="text-sm text-muted">No qualifying notes recorded.</li>
            )}
          </ul>
          {setup.reasonsAgainst.length > 0 && (
            <ul className="space-y-1.5 border-t border-border pt-3">
              {setup.reasonsAgainst.map((r, i) => (
                <li key={`against-${i}`} className="flex items-start gap-2 text-xs text-warn">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={2} />
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>

      {/* (c) Confidence breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Confidence Breakdown</CardTitle>
        </CardHeader>
        <CardBody>
          <ConfidenceBreakdown setup={setup} />
        </CardBody>
      </Card>

      {/* (d) Execution plan */}
      <Card>
        <CardHeader>
          <CardTitle>Execution Plan</CardTitle>
        </CardHeader>
        <CardBody className="space-y-4">
          <ExecutionPlanCard setup={setup} />
          <SetupActions setup={setup} />
        </CardBody>
      </Card>

      {/* (e) Key stats */}
      <Card>
        <CardHeader>
          <CardTitle>Key Stats</CardTitle>
        </CardHeader>
        <CardBody>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
            {keyStats.map((s) => (
              <div key={s.label} className="min-w-0">
                <dt className="text-2xs font-medium uppercase tracking-widest text-faint">{s.label}</dt>
                <dd className={cn("mt-0.5 truncate font-mono text-xs font-semibold tnum text-text")}>{s.value}</dd>
              </div>
            ))}
          </dl>
        </CardBody>
      </Card>
    </div>
  );
}
