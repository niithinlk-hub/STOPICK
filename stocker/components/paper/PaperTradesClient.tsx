"use client";

import { useMemo, useState } from "react";
import { Check, Plus, RefreshCw, Trash2, Wallet, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, CardBody, GradePill, Stat } from "@/components/ui/primitives";
import { fmtNumber, fmtPct, fmtPrice } from "@/lib/format";
import { summarize, tradeMetrics, usePaperTrades, type PaperTrade } from "@/lib/client/store";
import { useQuotes } from "@/lib/client/useQuotes";
import { useAuth } from "@/components/auth/AuthProvider";
import { SignInNotice } from "@/components/auth/SignInNotice";
import { DataTools } from "@/components/portfolio/DataTools";
import type { Market } from "@/lib/engine/types";

const MARKETS: Market[] = ["US", "NSE"];

const moneyTone = (v: number | null | undefined) =>
  v == null ? "text-faint" : v > 0 ? "text-bull" : v < 0 ? "text-bear" : "text-muted";

function signedMoney(v: number | null, market: Market): string {
  if (v == null) return "—";
  const s = fmtPrice(Math.abs(v), market);
  return v < 0 ? `-${s}` : `+${s}`;
}

export function PaperTradesClient() {
  const { user, loading: authLoading } = useAuth();
  const { trades, open, close, remove, hydrated } = usePaperTrades();
  const [showForm, setShowForm] = useState(false);

  const openTrades = useMemo(() => trades.filter((t) => t.status === "open"), [trades]);
  const closedTrades = useMemo(() => trades.filter((t) => t.status === "closed"), [trades]);

  const targets = useMemo(() => openTrades.map((t) => ({ ticker: t.ticker, market: t.market })), [openTrades]);
  const { quotes, loading, refresh, refreshedAt } = useQuotes(targets);

  const priceMap = useMemo(() => {
    const m: Record<string, number> = {};
    for (const [k, v] of Object.entries(quotes)) m[k] = v.price;
    return m;
  }, [quotes]);

  const summary = useMemo(() => summarize(trades, priceMap), [trades, priceMap]);

  const onClose = (t: PaperTrade) => {
    const live = quotes[`${t.ticker}|${t.market}`]?.price ?? null;
    const raw = window.prompt(`Close ${t.ticker} at price:`, live != null ? String(live) : "");
    if (raw == null) return;
    const exit = Number(raw);
    if (!Number.isFinite(exit) || exit <= 0) {
      window.alert("Enter a valid exit price.");
      return;
    }
    close(t.id, exit);
  };

  if (!authLoading && !user) {
    return (
      <div className="mx-auto max-w-[1400px] px-4 py-5 sm:px-6">
        <header className="mb-5">
          <h1 className="text-lg font-semibold tracking-tight">Paper Trades</h1>
        </header>
        <SignInNotice feature="paper-trading blotter" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-5 sm:px-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Paper Trades</h1>
          <p className="mt-0.5 text-sm text-muted">
            Log scanner setups and mark them to market — see whether the signals actually paid off. Stored in this browser only.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={refresh} disabled={!openTrades.length || loading} aria-label="Refresh prices">
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} strokeWidth={2} />
            Refresh
          </Button>
          <Button onClick={() => setShowForm((v) => !v)}>
            {showForm ? <X className="h-4 w-4" strokeWidth={2} /> : <Plus className="h-4 w-4" strokeWidth={2} />}
            {showForm ? "Cancel" : "Add trade"}
          </Button>
          <DataTools />
        </div>
      </header>

      {showForm && <ManualTradeForm onAdd={open} onDone={() => setShowForm(false)} />}

      {/* Scoreboard */}
      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Open" value={fmtNumber(summary.openCount, 0)} hint={`${summary.closedCount} closed`} />
        <Stat
          label="Win Rate"
          value={summary.winRate == null ? "—" : `${summary.winRate.toFixed(0)}%`}
          hint={`${summary.wins}W / ${summary.losses}L`}
        />
        <Stat label="Avg R" value={<span className={moneyTone(summary.avgR)}>{summary.avgR == null ? "—" : `${fmtNumber(summary.avgR, 2)}R`}</span>} />
        <Stat
          label="Profit Factor"
          value={summary.profitFactor == null ? "—" : summary.profitFactor === Infinity ? "∞" : fmtNumber(summary.profitFactor, 2)}
        />
        <Stat label="Realized" value={<span className={moneyTone(summary.realizedPnl)}>{signedMoneyMixed(summary.realizedPnl, trades)}</span>} />
        <Stat
          label="Unrealized"
          value={<span className={moneyTone(summary.unrealizedPnl)}>{signedMoneyMixed(summary.unrealizedPnl, trades)}</span>}
          hint={loading ? "marking…" : refreshedAt ? "marked-to-market" : undefined}
        />
      </div>

      {!hydrated ? null : trades.length === 0 ? (
        <Card>
          <CardBody className="flex flex-col items-center gap-3 py-16 text-center">
            <span className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 ring-1 ring-brand/20">
              <Wallet className="h-6 w-6 text-brand" strokeWidth={1.75} />
            </span>
            <p className="text-sm text-muted">
              No paper trades yet. Hit <span className="text-text">Paper trade this</span> on any scanner setup, or add one manually.
            </p>
          </CardBody>
        </Card>
      ) : (
        <div className="space-y-6">
          {/* Open positions */}
          <section>
            <h2 className="mb-2 text-sm font-semibold text-text">Open positions</h2>
            {openTrades.length === 0 ? (
              <p className="text-sm text-muted">No open positions.</p>
            ) : (
              <div className="overflow-x-auto rounded-xl border border-border bg-surface shadow-card">
                <table className="w-full border-collapse text-sm">
                  <thead className="bg-elevated">
                    <tr className="border-b border-border text-2xs font-semibold uppercase tracking-wider text-faint">
                      <th className="px-3 py-2.5 text-left">Symbol</th>
                      <th className="px-3 py-2.5 text-right">Qty</th>
                      <th className="px-3 py-2.5 text-right">Entry</th>
                      <th className="px-3 py-2.5 text-right">Stop</th>
                      <th className="px-3 py-2.5 text-right">Target</th>
                      <th className="px-3 py-2.5 text-right">Mark</th>
                      <th className="px-3 py-2.5 text-right">P&amp;L</th>
                      <th className="px-3 py-2.5 text-right">R</th>
                      <th className="px-3 py-2.5 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {openTrades.map((t) => {
                      const live = quotes[`${t.ticker}|${t.market}`]?.price ?? null;
                      const m = tradeMetrics(t, live);
                      return (
                        <tr key={t.id} className="border-b border-border/60 odd:bg-white/[0.015]">
                          <td className="whitespace-nowrap px-3 py-2.5">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-sm font-bold text-text tnum">{t.ticker}</span>
                              <Badge tone="neutral">{t.market}</Badge>
                              {t.grade && <GradePill grade={t.grade} />}
                            </div>
                            <div className="mt-0.5 text-2xs text-faint">
                              {t.source === "scan" ? `${t.setupFamily ?? "setup"} · ${t.pattern ?? "—"}` : "manual"} · {new Date(t.openedAt).toLocaleDateString()}
                            </div>
                          </td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-muted tnum">{fmtNumber(t.qty, 0)}</td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-text tnum">{fmtPrice(t.entry, t.market)}</td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-faint tnum">{t.stop == null ? "—" : fmtPrice(t.stop, t.market)}</td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-faint tnum">{t.target == null ? "—" : fmtPrice(t.target, t.market)}</td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-text tnum">{live == null ? (loading ? "…" : "—") : fmtPrice(live, t.market)}</td>
                          <td className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", moneyTone(m.pnl))}>
                            {m.pnl == null ? "—" : (
                              <span>
                                {signedMoney(m.pnl, t.market)}
                                <span className="ml-1 text-2xs text-faint">{m.pnlPct == null ? "" : fmtPct(m.pnlPct)}</span>
                              </span>
                            )}
                          </td>
                          <td className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", moneyTone(m.r))}>
                            {m.r == null ? "—" : `${fmtNumber(m.r, 2)}R`}
                          </td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right">
                            <div className="inline-flex items-center gap-1">
                              <button
                                type="button"
                                onClick={() => onClose(t)}
                                className="rounded-md px-2 py-1 text-2xs font-medium text-muted transition-colors hover:bg-white/[0.05] hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
                              >
                                Close
                              </button>
                              <button
                                type="button"
                                onClick={() => remove(t.id)}
                                aria-label={`Delete ${t.ticker}`}
                                className="rounded-md p-1.5 text-muted transition-colors hover:bg-bear/10 hover:text-bear focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
                              >
                                <Trash2 className="h-4 w-4" strokeWidth={2} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* Closed trades */}
          {closedTrades.length > 0 && (
            <section>
              <h2 className="mb-2 text-sm font-semibold text-text">Closed trades</h2>
              <div className="overflow-x-auto rounded-xl border border-border bg-surface shadow-card">
                <table className="w-full border-collapse text-sm">
                  <thead className="bg-elevated">
                    <tr className="border-b border-border text-2xs font-semibold uppercase tracking-wider text-faint">
                      <th className="px-3 py-2.5 text-left">Symbol</th>
                      <th className="px-3 py-2.5 text-right">Held</th>
                      <th className="px-3 py-2.5 text-right">Entry → Exit</th>
                      <th className="px-3 py-2.5 text-right">P&amp;L</th>
                      <th className="px-3 py-2.5 text-right">R</th>
                      <th className="px-3 py-2.5 text-center">Result</th>
                      <th className="px-3 py-2.5 text-right" />
                    </tr>
                  </thead>
                  <tbody>
                    {closedTrades.map((t) => {
                      const m = tradeMetrics(t, null);
                      const held = t.closedAt ? Math.max(0, Math.round((t.closedAt - t.openedAt) / 86_400_000)) : null;
                      const win = (m.pnl ?? 0) >= 0;
                      return (
                        <tr key={t.id} className="border-b border-border/60 odd:bg-white/[0.015]">
                          <td className="whitespace-nowrap px-3 py-2.5">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-sm font-bold text-text tnum">{t.ticker}</span>
                              <Badge tone="neutral">{t.market}</Badge>
                            </div>
                          </td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-faint tnum">{held == null ? "—" : `${held}d`}</td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-muted tnum">
                            {fmtPrice(t.entry, t.market)} → {t.exit == null ? "—" : fmtPrice(t.exit, t.market)}
                          </td>
                          <td className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", moneyTone(m.pnl))}>
                            {m.pnl == null ? "—" : (
                              <span>
                                {signedMoney(m.pnl, t.market)}
                                <span className="ml-1 text-2xs text-faint">{m.pnlPct == null ? "" : fmtPct(m.pnlPct)}</span>
                              </span>
                            )}
                          </td>
                          <td className={cn("whitespace-nowrap px-3 py-2.5 text-right font-mono tnum", moneyTone(m.r))}>
                            {m.r == null ? "—" : `${fmtNumber(m.r, 2)}R`}
                          </td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-center">
                            <Badge tone={win ? "bull" : "bear"}>{win ? "Win" : "Loss"}</Badge>
                          </td>
                          <td className="whitespace-nowrap px-3 py-2.5 text-right">
                            <button
                              type="button"
                              onClick={() => remove(t.id)}
                              aria-label={`Delete ${t.ticker}`}
                              className="rounded-md p-1.5 text-muted transition-colors hover:bg-bear/10 hover:text-bear focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
                            >
                              <Trash2 className="h-4 w-4" strokeWidth={2} />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

/* ----------------------------- manual entry ------------------------------ */

function ManualTradeForm({
  onAdd,
  onDone,
}: {
  onAdd: (t: Omit<PaperTrade, "id" | "openedAt" | "status">) => void;
  onDone: () => void;
}) {
  const [ticker, setTicker] = useState("");
  const [market, setMarket] = useState<Market>("US");
  const [qty, setQty] = useState("");
  const [entry, setEntry] = useState("");
  const [stop, setStop] = useState("");
  const [target, setTarget] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = ticker.trim().toUpperCase();
    const q = Number(qty);
    const e0 = Number(entry);
    if (!t || !Number.isFinite(q) || q <= 0 || !Number.isFinite(e0) || e0 <= 0) {
      window.alert("Ticker, a positive quantity, and a positive entry are required.");
      return;
    }
    onAdd({
      ticker: t,
      market,
      side: "long",
      qty: Math.round(q),
      entry: e0,
      stop: stop.trim() ? Number(stop) : null,
      target: target.trim() ? Number(target) : null,
      source: "manual",
    });
    onDone();
  };

  const field = "w-full rounded-lg border border-border bg-elevated/60 px-3 py-2 font-mono text-sm text-text tnum placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";
  const lbl = "text-2xs font-medium uppercase tracking-widest text-faint";

  return (
    <Card className="mb-5">
      <CardBody>
        <form onSubmit={submit} className="grid grid-cols-2 items-end gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <div className="space-y-1.5">
            <label htmlFor="pt-ticker" className={lbl}>Ticker</label>
            <input id="pt-ticker" value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="AAPL" className={cn(field, "uppercase")} />
          </div>
          <div className="space-y-1.5">
            <span className={lbl}>Market</span>
            <div className="flex rounded-lg border border-border bg-elevated/60 p-0.5">
              {MARKETS.map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMarket(m)}
                  aria-pressed={market === m}
                  className={cn(
                    "flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
                    market === m ? "bg-brand/15 text-text ring-1 ring-inset ring-brand/30" : "text-muted hover:text-text",
                  )}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-1.5">
            <label htmlFor="pt-qty" className={lbl}>Qty</label>
            <input id="pt-qty" value={qty} onChange={(e) => setQty(e.target.value)} inputMode="numeric" placeholder="100" className={field} />
          </div>
          <div className="space-y-1.5">
            <label htmlFor="pt-entry" className={lbl}>Entry</label>
            <input id="pt-entry" value={entry} onChange={(e) => setEntry(e.target.value)} inputMode="decimal" placeholder="0.00" className={field} />
          </div>
          <div className="space-y-1.5">
            <label htmlFor="pt-stop" className={lbl}>Stop</label>
            <input id="pt-stop" value={stop} onChange={(e) => setStop(e.target.value)} inputMode="decimal" placeholder="optional" className={field} />
          </div>
          <div className="space-y-1.5">
            <label htmlFor="pt-target" className={lbl}>Target</label>
            <input id="pt-target" value={target} onChange={(e) => setTarget(e.target.value)} inputMode="decimal" placeholder="optional" className={field} />
          </div>
          <div className="col-span-2 sm:col-span-3 lg:col-span-6">
            <Button type="submit">
              <Check className="h-4 w-4" strokeWidth={2} />
              Log trade
            </Button>
          </div>
        </form>
      </CardBody>
    </Card>
  );
}

/** Realized/unrealized totals can mix ₹ and $; show with the dominant market's symbol. */
function signedMoneyMixed(v: number, trades: PaperTrade[]): string {
  const market: Market = trades.some((t) => t.market === "NSE") && !trades.some((t) => t.market === "US") ? "NSE" : "US";
  return signedMoney(v, market);
}
