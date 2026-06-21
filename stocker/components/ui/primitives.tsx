"use client";

import { forwardRef } from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { cn } from "@/lib/cn";
import { DIRECTION_STYLES, GRADE_STYLES, scoreColor, scoreRgb } from "@/lib/grades";
import type { Direction, Grade } from "@/lib/engine/types";

/* -------------------------------- Card --------------------------------- */

export function Card({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("rounded-xl border border-border bg-surface shadow-card", className)} {...props}>
      {children}
    </div>
  );
}

export function CardHeader({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("flex items-center justify-between gap-3 border-b border-border px-4 py-3", className)}>{children}</div>;
}

export function CardTitle({ className, children }: { className?: string; children: React.ReactNode }) {
  return <h3 className={cn("text-sm font-semibold tracking-tight text-text", className)}>{children}</h3>;
}

export function CardBody({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("p-4", className)}>{children}</div>;
}

/* ------------------------------- Badge --------------------------------- */

type BadgeTone = "neutral" | "bull" | "bear" | "warn" | "info" | "brand";
const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "bg-white/[0.04] text-muted ring-border",
  bull: "bg-bull/10 text-bull ring-bull/30",
  bear: "bg-bear/10 text-bear ring-bear/30",
  warn: "bg-warn/10 text-warn ring-warn/30",
  info: "bg-info/10 text-info ring-info/30",
  brand: "bg-brand/10 text-brand ring-brand/30",
};

export function Badge({
  tone = "neutral",
  className,
  children,
}: {
  tone?: BadgeTone;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-2xs font-medium ring-1 ring-inset",
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ------------------------------ GradePill ------------------------------ */

export function GradePill({ grade, className }: { grade: Grade; className?: string }) {
  const s = GRADE_STYLES[grade];
  return (
    <span
      className={cn(
        "inline-flex min-w-[2.25rem] items-center justify-center rounded-md px-2 py-0.5 font-mono text-xs font-bold ring-1 ring-inset tnum",
        s.text,
        s.bg,
        s.ring,
        className,
      )}
      aria-label={`Grade ${s.label}`}
    >
      {s.label}
    </span>
  );
}

/* ----------------------------- DirectionTag ---------------------------- */

export function DirectionTag({ direction, className }: { direction: Direction; className?: string }) {
  const s = DIRECTION_STYLES[direction];
  const Icon = direction === "bullish" ? ArrowUpRight : direction === "bearish" ? ArrowDownRight : Minus;
  return (
    <span className={cn("inline-flex items-center gap-1 text-xs font-medium", s.text, className)}>
      <Icon className="h-3.5 w-3.5" strokeWidth={2.5} />
      {s.label}
    </span>
  );
}

/* ------------------------------- Button -------------------------------- */

type ButtonVariant = "primary" | "secondary" | "ghost";
const BTN: Record<ButtonVariant, string> = {
  primary:
    "bg-gradient-to-b from-brand to-brand-2 text-bg font-semibold hover:brightness-110 shadow-glow disabled:from-faint disabled:to-faint disabled:shadow-none",
  secondary: "border border-border bg-elevated text-text hover:bg-overlay",
  ghost: "text-muted hover:bg-white/[0.05] hover:text-text",
};

export const Button = forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }>(
  ({ variant = "primary", className, children, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm transition-all",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
        "disabled:cursor-not-allowed disabled:opacity-60",
        BTN[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  ),
);
Button.displayName = "Button";

/* ------------------------------ Skeleton ------------------------------- */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton rounded-md", className)} />;
}

/* -------------------------------- Stat --------------------------------- */

export function Stat({
  label,
  value,
  delta,
  hint,
  className,
}: {
  label: string;
  value: React.ReactNode;
  delta?: { value: number; suffix?: string };
  hint?: string;
  className?: string;
}) {
  return (
    <Card className={cn("p-4", className)}>
      <div className="text-2xs font-medium uppercase tracking-widest text-faint">{label}</div>
      <div className="mt-1.5 font-mono text-2xl font-semibold tracking-tight tnum">{value}</div>
      <div className="mt-1 flex items-center gap-2">
        {delta && (
          <span className={cn("inline-flex items-center gap-0.5 font-mono text-xs tnum", delta.value >= 0 ? "text-bull" : "text-bear")}>
            {delta.value >= 0 ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
            {Math.abs(delta.value).toFixed(2)}
            {delta.suffix ?? ""}
          </span>
        )}
        {hint && <span className="text-2xs text-faint">{hint}</span>}
      </div>
    </Card>
  );
}

/* ----------------------------- ScoreMeter ------------------------------ */

/** Horizontal 0–100 score meter with color ramp. */
export function ScoreMeter({ score, className }: { score: number; className?: string }) {
  const pct = Math.max(0, Math.min(100, score));
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: scoreRgb(score) }} />
      </div>
      <span className={cn("w-10 shrink-0 text-right font-mono text-xs font-semibold tnum", scoreColor(score))}>{score.toFixed(0)}</span>
    </div>
  );
}

/** Radial score gauge (SVG). size in px. */
export function ScoreGauge({ score, size = 88 }: { score: number; size?: number }) {
  const pct = Math.max(0, Math.min(100, score));
  const stroke = 8;
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(var(--elevated))" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={scoreRgb(score)}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className={cn("font-mono text-xl font-bold tnum", scoreColor(score))}>{score.toFixed(0)}</span>
        <span className="text-2xs uppercase tracking-wide text-faint">score</span>
      </div>
    </div>
  );
}
