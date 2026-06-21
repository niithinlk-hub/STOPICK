"use client";

import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/cn";
import { fmtPrice, fmtCompact, fmtNumber } from "@/lib/format";
import { Badge } from "@/components/ui/primitives";
import type { SetupSignal } from "@/lib/engine/types";

/**
 * Execution plan spec grid: entry / stop / targets / R:R / sizing / capital risk,
 * plus a stop→entry→1R→2R→3R ladder and any risk warnings.
 */
export function ExecutionPlanCard({ setup }: { setup: SetupSignal }) {
  const plan = setup.executionPlan;
  const market = setup.market;

  if (!plan) {
    return <p className="text-sm text-muted">No execution plan — this setup did not produce a tradable entry.</p>;
  }

  const usingAtrStop = Math.abs(plan.stop - plan.atrStop) <= Math.abs(plan.stop - plan.structureStop);
  const stopNote = usingAtrStop
    ? `ATR stop (struct ${fmtPrice(plan.structureStop, market)})`
    : `Structure stop (ATR ${fmtPrice(plan.atrStop, market)})`;

  const Spec = ({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) => (
    <div className="rounded-lg border border-border bg-elevated/40 px-3 py-2">
      <div className="text-2xs font-medium uppercase tracking-widest text-faint">{label}</div>
      <div className="mt-0.5 font-mono text-sm font-semibold tnum text-text">{value}</div>
      {sub && <div className="mt-0.5 text-2xs text-faint">{sub}</div>}
    </div>
  );

  // Ladder geometry: map prices onto a 0–100 track between min and max.
  const points = [
    { key: "stop", label: "Stop", price: plan.stop, color: "rgb(var(--bear))" },
    { key: "entry", label: "Entry", price: plan.entry, color: "rgb(var(--info))" },
    { key: "1r", label: "1R", price: plan.target1r, color: "rgb(var(--bull))" },
    { key: "2r", label: "2R", price: plan.target2r, color: "rgb(var(--bull))" },
    { key: "3r", label: "3R", price: plan.target3r, color: "rgb(var(--bull))" },
  ].filter((p) => Number.isFinite(p.price));
  const prices = points.map((p) => p.price);
  const lo = Math.min(...prices);
  const hi = Math.max(...prices);
  const span = hi - lo || 1;
  const posPct = (price: number) => ((price - lo) / span) * 100;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        <Spec label="Entry" value={fmtPrice(plan.entry, market)} />
        <Spec label="Stop" value={fmtPrice(plan.stop, market)} sub={stopNote} />
        <Spec label="R : R" value={<Badge tone={plan.rrRatio >= 2 ? "bull" : "info"}>{fmtNumber(plan.rrRatio, 2)}</Badge>} />
        <Spec label="Target 1R" value={fmtPrice(plan.target1r, market)} />
        <Spec label="Target 2R" value={fmtPrice(plan.target2r, market)} />
        <Spec label="Target 3R" value={fmtPrice(plan.target3r, market)} />
        <Spec label="Position Size" value={`${fmtCompact(plan.positionSizeShares)} sh`} />
        <Spec label="Position Value" value={fmtPrice(plan.positionValue, market)} sub={fmtCompact(plan.positionValue)} />
        <Spec label="Capital Risk" value={fmtPrice(plan.capitalRisk, market)} sub={fmtCompact(plan.capitalRisk)} />
      </div>

      {/* Ladder: stop -> entry -> 1R -> 2R -> 3R */}
      <div className="pt-1">
        <div className="relative mx-1 h-px bg-border-strong">
          {points.map((p) => (
            <div
              key={p.key}
              className="absolute top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center"
              style={{ left: `${posPct(p.price)}%` }}
            >
              <span className="h-2.5 w-2.5 rounded-full ring-2 ring-surface" style={{ background: p.color }} />
              <span className="mt-1 font-mono text-2xs tnum text-faint">{p.label}</span>
            </div>
          ))}
        </div>
        <div className="h-7" aria-hidden />
      </div>

      {plan.warnings.length > 0 && (
        <ul className="space-y-1.5">
          {plan.warnings.map((w, i) => (
            <li key={i} className={cn("flex items-start gap-2 text-xs text-warn")}>
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={2} />
              <span>{w}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
