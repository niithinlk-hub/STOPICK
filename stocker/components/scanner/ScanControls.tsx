"use client";

import { useId } from "react";
import { Loader2, Play } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";
import type { Country, ScanParams, SetupMode, Timeframe, UniverseSource } from "@/lib/engine/types";

/* ----------------------------- Segmented ------------------------------- */

interface SegOption<T extends string> {
  value: T;
  label: string;
}

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string;
  value: T;
  options: SegOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <div className="text-2xs font-medium uppercase tracking-widest text-faint">{label}</div>
      <div
        role="group"
        aria-label={label}
        className="flex flex-wrap gap-1 rounded-lg border border-border bg-elevated/60 p-1"
      >
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              aria-pressed={active}
              disabled={disabled}
              onClick={() => onChange(opt.value)}
              className={cn(
                "flex-1 whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
                "disabled:cursor-not-allowed disabled:opacity-60",
                active
                  ? "bg-brand/15 text-text ring-1 ring-brand/30"
                  : "text-muted hover:bg-white/[0.04] hover:text-text",
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

/* ---------------------------- ScanControls ----------------------------- */

const COUNTRY_OPTS: SegOption<Country>[] = [
  { value: "US", label: "US" },
  { value: "NSE", label: "NSE" },
  { value: "BOTH", label: "Both" },
];

const SOURCE_OPTS: SegOption<UniverseSource>[] = [
  { value: "sample", label: "Sample 15" },
  { value: "tier_1", label: "Set 1 · 500" },
  { value: "tier_2", label: "Set 2 · 500" },
  { value: "manual", label: "Manual" },
];

const TIMEFRAME_OPTS: SegOption<Timeframe>[] = [
  { value: "1d", label: "1D" },
  { value: "4h", label: "4H" },
  { value: "1h", label: "1H" },
  { value: "15m", label: "15m" },
];

const MODE_OPTS: SegOption<SetupMode>[] = [
  { value: "breakout", label: "Breakout" },
  { value: "pullback", label: "Pullback" },
  { value: "both", label: "Both" },
];

export function ScanControls({
  params,
  onChange,
  onRun,
  loading,
}: {
  params: ScanParams;
  onChange: (params: ScanParams) => void;
  onRun: () => void;
  loading: boolean;
}) {
  const sliderId = useId();
  const manualId = useId();

  const set = <K extends keyof ScanParams>(key: K, value: ScanParams[K]) =>
    onChange({ ...params, [key]: value });

  // tier_3 is an NSE-only breadth set — only offer it when NSE is in scope.
  const sourceOpts: SegOption<UniverseSource>[] =
    params.country === "US"
      ? SOURCE_OPTS
      : [...SOURCE_OPTS.slice(0, 3), { value: "tier_3", label: "Set 3 · NSE" }, ...SOURCE_OPTS.slice(3)];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Scan Controls</CardTitle>
      </CardHeader>
      <CardBody className="space-y-5">
        <Segmented
          label="Country"
          value={params.country}
          options={COUNTRY_OPTS}
          onChange={(v) => set("country", v)}
          disabled={loading}
        />
        <Segmented
          label="Universe Source"
          value={params.source}
          options={sourceOpts}
          onChange={(v) => set("source", v)}
          disabled={loading}
        />
        <Segmented
          label="Timeframe"
          value={params.timeframe}
          options={TIMEFRAME_OPTS}
          onChange={(v) => set("timeframe", v)}
          disabled={loading}
        />
        <Segmented
          label="Setup Mode"
          value={params.setupMode}
          options={MODE_OPTS}
          onChange={(v) => set("setupMode", v)}
          disabled={loading}
        />

        {/* Min score slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label htmlFor={sliderId} className="text-2xs font-medium uppercase tracking-widest text-faint">
              Minimum Score
            </label>
            <span className="font-mono text-sm font-semibold text-brand tnum">{params.minScore}</span>
          </div>
          <input
            id={sliderId}
            type="range"
            min={0}
            max={100}
            step={1}
            value={params.minScore}
            disabled={loading}
            onChange={(e) => set("minScore", Number(e.target.value))}
            aria-valuetext={`${params.minScore} of 100`}
            className={cn(
              "h-1.5 w-full cursor-pointer appearance-none rounded-full bg-elevated accent-brand",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
              "disabled:cursor-not-allowed disabled:opacity-60",
            )}
            style={{
              background: `linear-gradient(to right, rgb(var(--brand)) 0%, rgb(var(--brand)) ${params.minScore}%, rgb(var(--elevated)) ${params.minScore}%, rgb(var(--elevated)) 100%)`,
            }}
          />
          <div className="flex justify-between text-2xs text-faint tnum">
            <span>0</span>
            <span>100</span>
          </div>
        </div>

        {/* Manual symbols — only when source is manual */}
        {params.source === "manual" && (
          <div className="space-y-1.5">
            <label htmlFor={manualId} className="text-2xs font-medium uppercase tracking-widest text-faint">
              Manual Symbols
            </label>
            <textarea
              id={manualId}
              rows={3}
              value={params.manualSymbols ?? ""}
              disabled={loading}
              onChange={(e) => set("manualSymbols", e.target.value)}
              placeholder="AAPL, MSFT, NVDA, RELIANCE.NS"
              className={cn(
                "w-full resize-y rounded-lg border border-border bg-elevated/60 px-3 py-2 font-mono text-xs text-text tnum",
                "placeholder:text-faint",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
                "disabled:cursor-not-allowed disabled:opacity-60",
              )}
            />
            <p className="text-2xs leading-relaxed text-faint">
              Comma or space separated. Use the provider ticker (append <span className="font-mono">.NS</span> for NSE).
            </p>
          </div>
        )}

        <Button
          type="button"
          onClick={onRun}
          disabled={loading}
          className="w-full"
          aria-busy={loading}
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} aria-hidden />
              Scanning…
            </>
          ) : (
            <>
              <Play className="h-4 w-4" strokeWidth={2} aria-hidden />
              Run scan
            </>
          )}
        </Button>
      </CardBody>
    </Card>
  );
}
