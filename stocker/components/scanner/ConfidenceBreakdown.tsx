"use client";

import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/cn";
import { CONFIG, COMPONENT_LABELS } from "@/lib/config";
import { scoreRgb } from "@/lib/grades";
import { GradePill } from "@/components/ui/primitives";
import type { SetupSignal } from "@/lib/engine/types";

/**
 * Renders setup.breakdown as horizontal segmented bars, one per component, with
 * each component's profile weight shown as a small chip. Rows sort by
 * value*weight descending so the highest-contribution factors lead.
 */
export function ConfidenceBreakdown({ setup }: { setup: SetupSignal }) {
  const reduce = useReducedMotion();

  const profileKey = setup.setupFamily === "pullback" ? "bullish_pullback" : "bullish_breakout";
  const weights = CONFIG.scoringProfiles[profileKey]?.weights ?? {};

  const rows = Object.entries(setup.breakdown)
    .map(([key, value]) => {
      const weight = weights[key] ?? 0;
      return {
        key,
        label: COMPONENT_LABELS[key] ?? key,
        value: Number.isFinite(value) ? value : 0,
        weight,
        contribution: (Number.isFinite(value) ? value : 0) * weight,
      };
    })
    .sort((a, b) => b.contribution - a.contribution);

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <span className="text-2xs font-medium uppercase tracking-widest text-faint">Blended score</span>
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-semibold tnum text-text">{setup.score.toFixed(0)}</span>
          <GradePill grade={setup.grade} />
        </div>
      </div>

      <div className="space-y-2">
        {rows.map((row, i) => {
          const pct = Math.max(0, Math.min(100, row.value));
          return (
            <div key={row.key} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1">
              <div className="flex min-w-0 items-center gap-2">
                <span className="truncate text-xs text-muted">{row.label}</span>
                <span className="shrink-0 rounded bg-white/[0.04] px-1 py-px font-mono text-2xs tnum text-faint ring-1 ring-inset ring-border">
                  w:{row.weight}
                </span>
              </div>
              <span className={cn("text-right font-mono text-xs font-semibold tnum text-text")}>{row.value.toFixed(0)}</span>
              <div className="col-span-2 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: scoreRgb(row.value) }}
                  initial={{ width: reduce ? `${pct}%` : 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: reduce ? 0 : 0.25, ease: "easeOut", delay: reduce ? 0 : i * 0.03 }}
                />
              </div>
            </div>
          );
        })}
        {rows.length === 0 && <p className="text-sm text-muted">No score breakdown available.</p>}
      </div>
    </div>
  );
}
