"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Plus, RefreshCw, Search, Star, Trash2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, CardBody } from "@/components/ui/primitives";
import { fmtPct, fmtPrice } from "@/lib/format";
import { useWatchlist } from "@/lib/client/store";
import { useQuotes } from "@/lib/client/useQuotes";
import { useAuth } from "@/components/auth/AuthProvider";
import { SignInNotice } from "@/components/auth/SignInNotice";
import { DataTools } from "@/components/portfolio/DataTools";
import type { Market } from "@/lib/engine/types";

const MARKETS: Market[] = ["US", "NSE"];

export function WatchlistClient() {
  const { user, loading: authLoading } = useAuth();
  const { items, add, remove, clear, hydrated } = useWatchlist();
  const [symbol, setSymbol] = useState("");
  const [market, setMarket] = useState<Market>("US");

  const targets = useMemo(() => items.map((i) => ({ ticker: i.symbol, market: i.market })), [items]);
  const { quotes, loading, refresh, refreshedAt } = useQuotes(targets);

  const byMarket = useMemo(() => {
    const m: Partial<Record<Market, string[]>> = {};
    for (const i of items) (m[i.market] ??= []).push(i.symbol);
    return m;
  }, [items]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const s = symbol.trim();
    if (!s) return;
    add({ symbol: s, market });
    setSymbol("");
  };

  const dayChange = (sym: string, mkt: Market): number | null => {
    const q = quotes[`${sym}|${mkt}`];
    if (!q || !q.prevClose) return null;
    return ((q.price - q.prevClose) / q.prevClose) * 100;
  };

  if (!authLoading && !user) {
    return (
      <div className="mx-auto max-w-[1100px] px-4 py-5 sm:px-6">
        <header className="mb-5">
          <h1 className="text-lg font-semibold tracking-tight">Watchlist</h1>
        </header>
        <SignInNotice feature="watchlist" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-5 sm:px-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Watchlist</h1>
          <p className="mt-0.5 text-sm text-muted">
            Track symbols and jump straight into a focused scan. Stored in this browser only.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={refresh} disabled={!items.length || loading} aria-label="Refresh prices">
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} strokeWidth={2} />
            Refresh
          </Button>
          <DataTools />
        </div>
      </header>

      {/* Add + bulk-scan controls */}
      <Card className="mb-5">
        <CardBody className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <form onSubmit={submit} className="flex flex-wrap items-end gap-2">
            <div className="space-y-1.5">
              <label htmlFor="wl-symbol" className="text-2xs font-medium uppercase tracking-widest text-faint">
                Symbol
              </label>
              <input
                id="wl-symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                placeholder="AAPL"
                className="w-36 rounded-lg border border-border bg-elevated/60 px-3 py-2 font-mono text-sm uppercase text-text tnum placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
              />
            </div>
            <div className="space-y-1.5">
              <span className="block text-2xs font-medium uppercase tracking-widest text-faint">Market</span>
              <div className="flex rounded-lg border border-border bg-elevated/60 p-0.5">
                {MARKETS.map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMarket(m)}
                    aria-pressed={market === m}
                    className={cn(
                      "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                      market === m ? "bg-brand/15 text-text ring-1 ring-inset ring-brand/30" : "text-muted hover:text-text",
                    )}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
            <Button type="submit" disabled={!symbol.trim()}>
              <Plus className="h-4 w-4" strokeWidth={2} />
              Add
            </Button>
          </form>

          {items.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {MARKETS.filter((m) => byMarket[m]?.length).map((m) => (
                <Link
                  key={m}
                  href={`/?symbols=${encodeURIComponent((byMarket[m] ?? []).join(","))}&country=${m}`}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-elevated px-3 py-2 text-xs font-medium text-text transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
                >
                  <Search className="h-3.5 w-3.5" strokeWidth={2} />
                  Scan {byMarket[m]?.length} {m}
                </Link>
              ))}
            </div>
          )}
        </CardBody>
      </Card>

      {/* List */}
      {!hydrated ? null : items.length === 0 ? (
        <Card>
          <CardBody className="flex flex-col items-center gap-3 py-16 text-center">
            <span className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 ring-1 ring-brand/20">
              <Star className="h-6 w-6 text-brand" strokeWidth={1.75} />
            </span>
            <p className="text-sm text-muted">
              No symbols yet. Add one above, or hit <span className="text-text">Add to watchlist</span> on any scanner setup.
            </p>
          </CardBody>
        </Card>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border bg-surface shadow-card">
          <table className="w-full border-collapse text-sm">
            <thead className="bg-elevated">
              <tr className="border-b border-border text-2xs font-semibold uppercase tracking-wider text-faint">
                <th scope="col" className="px-3 py-2.5 text-left">Symbol</th>
                <th scope="col" className="px-3 py-2.5 text-right">Price</th>
                <th scope="col" className="px-3 py-2.5 text-right">Day</th>
                <th scope="col" className="px-3 py-2.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => {
                const q = quotes[`${it.symbol}|${it.market}`];
                const chg = dayChange(it.symbol, it.market);
                return (
                  <tr key={it.id} className="border-b border-border/60 odd:bg-white/[0.015]">
                    <td className="whitespace-nowrap px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm font-bold text-text tnum">{it.symbol}</span>
                        <Badge tone="neutral">{it.market}</Badge>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-text tnum">
                      {q ? fmtPrice(q.price, it.market) : loading ? "…" : "—"}
                    </td>
                    <td
                      className={cn(
                        "whitespace-nowrap px-3 py-2.5 text-right font-mono tnum",
                        chg == null ? "text-faint" : chg >= 0 ? "text-bull" : "text-bear",
                      )}
                    >
                      {chg == null ? "—" : fmtPct(chg)}
                    </td>
                    <td className="whitespace-nowrap px-3 py-2.5 text-right">
                      <div className="inline-flex items-center gap-1">
                        <Link
                          href={`/?symbols=${encodeURIComponent(it.symbol)}&country=${it.market}`}
                          aria-label={`Scan ${it.symbol}`}
                          className="rounded-md p-1.5 text-muted transition-colors hover:bg-white/[0.05] hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
                        >
                          <Search className="h-4 w-4" strokeWidth={2} />
                        </Link>
                        <button
                          type="button"
                          onClick={() => remove(it.id)}
                          aria-label={`Remove ${it.symbol}`}
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

      {items.length > 0 && (
        <div className="mt-4 flex items-center justify-between text-2xs text-faint">
          <span>
            {items.length} symbol{items.length === 1 ? "" : "s"}
            {refreshedAt ? ` · prices as of ${new Date(refreshedAt).toLocaleTimeString()}` : ""}
          </span>
          <button type="button" onClick={clear} className="text-faint transition-colors hover:text-bear">
            Clear all
          </button>
        </div>
      )}
    </div>
  );
}
