# STOCKER — Design System (contract)

Dark-first **fintech trading workstation**. Audience: serious traders. Tone: precise,
high-trust, data-dense, calm. Reference feel: Linear × Bloomberg × TradingView.

## Tokens (Tailwind, defined in `app/globals.css`)

Use semantic classes only — never raw hex in components.

| Token | Use |
|-------|-----|
| `bg` | app background (deep blue-black `#070a0f`) |
| `surface` | panels / cards |
| `elevated` | nested cards, table headers |
| `overlay` | popovers, menus, modals |
| `border` / `border-strong` | hairlines / emphasized borders |
| `text` / `muted` / `faint` | primary / secondary / tertiary text |
| `brand` (cyan) / `brand-2` (indigo) | interactive, focus, gradients |
| `bull` (emerald) | up / bullish / positive |
| `bear` (red) | down / bearish / negative |
| `warn` (amber) | caution / event risk |
| `info` (sky) | neutral info |

Opacity utilities work: `bg-surface/70`, `border-border/60`, etc.

## Typography
- Sans: **Inter** (`font-sans`, via `next/font`). Mono: **JetBrains Mono** (`font-mono`).
- All numbers/prices/scores/columns use `font-mono` + `.tnum` (tabular figures) to prevent column jitter.
- Scale: 12 / 14 / 16 / 18 / 24 / 32. Body 14–16. Weight: headings 600–700, body 400, labels 500.

## Grades (badges/pills)
`A+` → bull, `A` → emerald/green, `B` → info (sky), `C` → warn (amber), `Reject` → faint/zinc.
Always pair color with the letter text (never color-only).

## Components & rules
- **Cards**: `bg-surface border border-border rounded-xl shadow-card`. Section padding 16/24.
- **Summary cards** (top of pages): label (xs, muted, uppercase tracking), big mono value, small delta with bull/bear color + arrow icon.
- **Tables**: sticky header (`bg-elevated`), zebra via `odd:bg-white/[0.015]`, row hover `hover:bg-white/[0.03]`, sortable headers with aria-sort, right-align numerics, tabular nums. Virtualize / cap rows if >200. Row click opens detail panel.
- **Score**: radial gauge OR horizontal meter, 0–100, color ramps faint→info→bull. Show grade pill beside it.
- **Confidence breakdown**: horizontal stacked/segmented bars per component (use `COMPONENT_LABELS`), weight-scaled.
- **Charts**: candlesticks via `lightweight-charts` (dark theme), with horizontal price lines for breakout level, retest zone, FVG zone, entry/stop/targets. Analytics (RS bars, regime, backtest equity/MC) via `recharts`.
- **Icons**: `lucide-react` only (no emoji), stroke 1.5–2, size tokens 16/20/24.
- **Motion** (`framer-motion`): 150–300ms, ease-out enter; stagger table rows 30ms; spring on cards; respect `prefers-reduced-motion`. Animate transform/opacity only.
- **Empty / loading / error**: skeleton shimmer (`.skeleton`) while scanning; meaningful empty states ("No setups cleared your filters"); error states with a retry action.
- **Accessibility**: focus rings (brand), ≥4.5:1 contrast, keyboard nav, aria-labels on icon buttons, aria-live on scan status.

## Layout
- Left **sidebar** nav (collapsible): Scanner, Top Setups, Relative Strength, Market Regime, Backtest. Active item highlighted (brand). Bottom: status / data source.
- Scanner page: left controls rail (country, universe source, timeframe, setup mode, min-score slider, manual symbols, weight-tuning accordion) → center results table → right/bottom **setup detail** drawer (why-qualified, score breakdown, execution plan, candlestick chart).
- `max-w` container on wide screens; responsive: rail collapses under table on < lg.

## Pages
1. **Scanner** — controls + summary cards + results table + detail panel.
2. **Top Setups** — ranked card grid of best A+/A setups.
3. **Relative Strength** — leaderboard sorted by RS score, sparkline-ish bars.
4. **Market Regime** — per-market regime cards (direction, vol state, breadth gauge).
5. **Backtest** — equity curve, walk-forward split, Monte Carlo distribution, trade table.
