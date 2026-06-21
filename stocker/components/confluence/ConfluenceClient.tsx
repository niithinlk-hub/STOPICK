"use client";

import { useCallback, useRef, useState } from "react";
import { AlertTriangle, GitMerge, Info, Play, RefreshCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, CardBody, ScoreMeter, Skeleton } from "@/components/ui/primitives";
import { fmtNumber, fmtPrice } from "@/lib/format";
import { runConfluence, type ConfluenceParams } from "@/lib/client/api";
import { QuickAddTrade } from "@/components/portfolio/QuickAddTrade";
import type { ConfluenceResponse, ConfluenceRow, Country, Timeframe } from "@/lib/engine/types";

type Side = "long" | "short" | "both";
type Source = "sample" | "tier_1" | "tier_2";

function Seg<T extends string>({ label, value, opts, onChange, disabled }: {
  label: string; value: T; opts: { v: T; l: string }[]; onChange: (v: T) => void; disabled?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <div className="text-2xs font-medium uppercase tracking-widest text-faint">{label}</div>
      <div className="flex flex-wrap gap-1 rounded-lg border border-border bg-elevated/60 p-1">
        {opts.map((o) => (
          <button
            key={o.v}
            type="button"
            disabled={disabled}
            aria-pressed={o.v === value}
            onClick={() => onChange(o.v)}
            className={cn(
              "flex-1 whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors disabled:opacity-60",
              o.v === value ? "bg-brand/15 text-text ring-1 ring-brand/30" : "text-muted hover:text-text",
            )}
          >
            {o.l}
          </button>
        ))}
      </div>
    </div>
  );
}

const sideTone = (s: string) => (s === "long" ? "bull" : s === "short" ? "bear" : "neutral");

function Row({ r }: { r: ConfluenceRow }) {
  const ccy = r.market;
  return (
    <tr className="border-b border-border/60 align-top odd:bg-white/[0.015]">
      <td className="whitespace-nowrap px-3 py-2.5">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-bold text-text tnum">{r.ticker}</span>
          <Badge tone="neutral">{r.market}</Badge>
        </div>
        <div className="mt-0.5 max-w-[22rem] text-2xs leading-snug text-faint">{r.reasons[0]}</div>
      </td>
      <td className="whitespace-nowrap px-3 py-2.5">
        <Badge tone={sideTone(r.side)} className="uppercase">{r.side}</Badge>
      </td>
      <td className="px-3 py-2.5">
        <div className="flex items-center gap-2">
          <ScoreMeter score={r.confidence} className="w-20" />
          <span className="w-8 text-right font-mono text-xs font-semibold tnum text-text">{r.confidence}</span>
        </div>
      </td>
      <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-text tnum">{fmtPrice(r.price, ccy)}</td>
      <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-muted tnum">
        {fmtNumber(r.stochK, 0)}/{fmtNumber(r.stochD, 0)}
        <span className="ml-1 text-2xs text-faint">{r.stochState}</span>
      </td>
      <td className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", r.rsiAbove50 ? "text-bull" : "text-bear")}>
        {fmtNumber(r.rsi, 0)}
      </td>
      <td className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", r.macdHist >= 0 ? "text-bull" : "text-bear")}>
        {fmtNumber(r.macdHist, 3)}
        {(r.macdCrossUp || r.macdCrossDown) && <span className="ml-1 text-2xs text-brand">✕</span>}
      </td>
      <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-text tnum">{r.entry == null ? "—" : fmtPrice(r.entry, ccy)}</td>
      <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-bear tnum">{r.stop == null ? "—" : fmtPrice(r.stop, ccy)}</td>
      <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-bull tnum">{r.target == null ? "—" : fmtPrice(r.target, ccy)}</td>
      <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-faint tnum">{r.stopPct == null ? "—" : `${fmtNumber(r.stopPct, 2)}%`}</td>
      <td className="whitespace-nowrap px-3 py-2.5 text-right">
        {r.side === "long" ? (
          <QuickAddTrade ticker={r.ticker} market={r.market} entry={r.entry} stop={r.stop} target={r.target} score={r.confidence} pattern="RSI Confluence" source="scan" />
        ) : (
          <span className="text-2xs text-faint">long-only</span>
        )}
      </td>
    </tr>
  );
}

export function ConfluenceClient() {
  const [country, setCountry] = useState<Country>("US");
  const [source, setSource] = useState<Source>("tier_1");
  const [timeframe, setTimeframe] = useState<Timeframe>("1d");
  const [side, setSide] = useState<Side>("long");
  const [data, setData] = useState<ConfluenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    try {
      const params: ConfluenceParams = { country, source, timeframe, side };
      const res = await runConfluence(params, ctrl.signal);
      if (!ctrl.signal.aborted) setData(res);
    } catch (err) {
      if (!ctrl.signal.aborted) setError(err instanceof Error ? err.message : "Scan failed.");
    } finally {
      if (abortRef.current === ctrl) setLoading(false);
    }
  }, [country, source, timeframe, side]);

  return (
    <div className="mx-auto max-w-[1500px] px-4 py-5 sm:px-6">
      <header className="mb-5">
        <h1 className="text-lg font-semibold tracking-tight">RSI Confluence</h1>
        <p className="mt-0.5 text-sm text-muted">
          Stochastic + RSI(50) + MACD-cross confluence. Long when Stochastic turns up from oversold, RSI &gt; 50, and MACD
          line crosses above signal — mirror for shorts. Stop at the swing level, 1.5R target.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <div className="space-y-4 lg:sticky lg:top-5 lg:self-start">
          <Card>
            <CardBody className="space-y-4">
              <Seg label="Country" value={country} onChange={setCountry}
                opts={[{ v: "US", l: "US" }, { v: "NSE", l: "NSE" }, { v: "BOTH", l: "Both" }]} disabled={loading} />
              <Seg label="Universe" value={source} onChange={setSource}
                opts={[{ v: "sample", l: "Sample" }, { v: "tier_1", l: "Set 1" }, { v: "tier_2", l: "Set 2" }]} disabled={loading} />
              <Seg label="Timeframe / Mode" value={timeframe} onChange={setTimeframe}
                opts={[{ v: "1d", l: "BTST·1D" }, { v: "1h", l: "1H" }, { v: "15m", l: "15m" }]} disabled={loading} />
              <Seg label="Direction" value={side} onChange={setSide}
                opts={[{ v: "long", l: "Long" }, { v: "short", l: "Short" }, { v: "both", l: "Both" }]} disabled={loading} />
              <Button type="button" onClick={run} disabled={loading} className="w-full">
                {loading ? <RefreshCw className="h-4 w-4 animate-spin" strokeWidth={2} /> : <Play className="h-4 w-4" strokeWidth={2} />}
                {loading ? "Scanning…" : "Run scan"}
              </Button>
              <p className="text-2xs leading-relaxed text-faint">
                BTST·1D = act on the next session after a confirmed daily close. Intraday modes screen 1H / 15m bars.
                Signals on the still-forming current bar can change — confirm on close.
              </p>
            </CardBody>
          </Card>
        </div>

        <div className="min-w-0 space-y-4">
          {data && (
            <Card className="border-info/30 bg-info/5">
              <CardBody className="flex items-start gap-2.5 py-3 text-xs text-muted">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-info" strokeWidth={2} />
                <div>
                  <span className="text-text">{data.signalCount}</span> {side} signal{data.signalCount === 1 ? "" : "s"} from {data.scannedSymbols} symbols · {data.mode}.
                  <ul className="mt-1 space-y-0.5 text-2xs text-faint">
                    {data.notes.map((n, i) => <li key={i}>• {n}</li>)}
                  </ul>
                </div>
              </CardBody>
            </Card>
          )}

          {error ? (
            <Card className="border-bear/30 bg-bear/5">
              <CardBody className="flex flex-col items-center gap-3 py-12 text-center">
                <AlertTriangle className="h-7 w-7 text-bear" strokeWidth={1.75} />
                <p className="text-sm text-muted">{error}</p>
                <Button variant="secondary" onClick={run}><RefreshCw className="h-4 w-4" strokeWidth={2} />Retry</Button>
              </CardBody>
            </Card>
          ) : loading ? (
            <Card><CardBody><Skeleton className="h-72 w-full" /></CardBody></Card>
          ) : !data ? (
            <Card>
              <CardBody className="flex flex-col items-center gap-3 py-16 text-center">
                <span className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 ring-1 ring-brand/20">
                  <GitMerge className="h-6 w-6 text-brand" strokeWidth={1.75} />
                </span>
                <p className="text-sm text-muted">Pick a universe + timeframe and run the confluence scan.</p>
              </CardBody>
            </Card>
          ) : data.rows.length === 0 ? (
            <Card>
              <CardBody className="py-16 text-center text-sm text-muted">
                No {side} confluence signals right now. Try the other direction, a different timeframe, or Set 2.
              </CardBody>
            </Card>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-border bg-surface shadow-card">
              <table className="w-full border-collapse text-sm">
                <thead className="bg-elevated">
                  <tr className="border-b border-border text-2xs font-semibold uppercase tracking-wider text-faint">
                    <th className="px-3 py-2.5 text-left">Symbol</th>
                    <th className="px-3 py-2.5 text-left">Side</th>
                    <th className="px-3 py-2.5 text-left">Confidence</th>
                    <th className="px-3 py-2.5 text-right">Price</th>
                    <th className="px-3 py-2.5 text-right">Stoch K/D</th>
                    <th className="px-3 py-2.5 text-right">RSI</th>
                    <th className="px-3 py-2.5 text-right">MACD hist</th>
                    <th className="px-3 py-2.5 text-right">Entry</th>
                    <th className="px-3 py-2.5 text-right">Stop</th>
                    <th className="px-3 py-2.5 text-right">Target</th>
                    <th className="px-3 py-2.5 text-right">Stop %</th>
                    <th className="px-3 py-2.5 text-right">Trade</th>
                  </tr>
                </thead>
                <tbody>{data.rows.map((r) => <Row key={`${r.ticker}|${r.market}`} r={r} />)}</tbody>
              </table>
            </div>
          )}

          <p className="text-2xs leading-relaxed text-faint">
            Three momentum oscillators off one price series — confluence reduces single-indicator whipsaw but adds lag and
            does not confirm a trend on its own. Research, not investment advice. Confirm on a closed bar and respect the stop.
          </p>
        </div>
      </div>
    </div>
  );
}
