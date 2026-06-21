"use client";

/**
 * Watchlist + paper-trading blotter, persisted in Supabase Postgres and scoped to
 * the signed-in user by row-level security. The hooks keep the same shape they had
 * under the old localStorage implementation (synchronous-looking `items`/`trades`
 * arrays + mutator callbacks) so the UI is unchanged; mutations apply optimistically
 * and then reconcile against the server.
 */
import { useCallback, useEffect, useState } from "react";
import { getSupabaseBrowser } from "@/lib/supabase/client";
import { useAuth } from "@/components/auth/AuthProvider";
import type { Grade, Market, SetupFamily } from "@/lib/engine/types";

export interface WatchItem {
  id: string;
  symbol: string; // display symbol (no .NS suffix)
  market: Market;
  note?: string;
  addedAt: number;
}

export interface PaperTrade {
  id: string;
  ticker: string;
  market: Market;
  side: "long";
  qty: number;
  entry: number;
  stop: number | null;
  target: number | null;
  openedAt: number;
  setupFamily?: SetupFamily;
  score?: number;
  grade?: Grade;
  pattern?: string;
  source: "scan" | "manual";
  status: "open" | "closed";
  exit?: number;
  closedAt?: number;
  note?: string;
}

/* ------------------------------- row mappers ----------------------------- */

type Row = Record<string, unknown>;
const num = (v: unknown): number => Number(v);
const numOrNull = (v: unknown): number | null => (v == null ? null : Number(v));

function mapWatchRow(r: Row): WatchItem {
  return {
    id: String(r.id),
    symbol: String(r.symbol),
    market: r.market as Market,
    note: (r.note as string | null) ?? undefined,
    addedAt: new Date(String(r.created_at)).getTime(),
  };
}

function mapTradeRow(r: Row): PaperTrade {
  return {
    id: String(r.id),
    ticker: String(r.ticker),
    market: r.market as Market,
    side: "long",
    qty: num(r.qty),
    entry: num(r.entry),
    stop: numOrNull(r.stop),
    target: numOrNull(r.target),
    openedAt: new Date(String(r.opened_at)).getTime(),
    setupFamily: (r.setup_family as SetupFamily | null) ?? undefined,
    score: r.score == null ? undefined : num(r.score),
    grade: (r.grade as Grade | null) ?? undefined,
    pattern: (r.pattern as string | null) ?? undefined,
    source: (r.source as "scan" | "manual") ?? "manual",
    status: (r.status as "open" | "closed") ?? "open",
    exit: r.exit == null ? undefined : num(r.exit),
    closedAt: r.closed_at ? new Date(String(r.closed_at)).getTime() : undefined,
    note: (r.note as string | null) ?? undefined,
  };
}

/* ------------------------------- watchlist ------------------------------- */

export function useWatchlist() {
  const supabase = getSupabaseBrowser();
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const [items, setItems] = useState<WatchItem[]>([]);
  const [hydrated, setHydrated] = useState(false);

  const refresh = useCallback(async () => {
    if (!userId) {
      setItems([]);
      setHydrated(true);
      return;
    }
    const { data } = await supabase
      .from("stocker_watchlist")
      .select("*")
      .order("created_at", { ascending: false });
    setItems((data ?? []).map(mapWatchRow));
    setHydrated(true);
  }, [supabase, userId]);

  useEffect(() => {
    setHydrated(false);
    refresh();
  }, [refresh]);

  const add = useCallback(
    async (entry: { symbol: string; market: Market; note?: string }) => {
      if (!userId) return;
      const symbol = entry.symbol.trim().toUpperCase();
      if (!symbol) return;
      setItems((prev) =>
        prev.some((p) => p.symbol === symbol && p.market === entry.market)
          ? prev
          : [{ id: `tmp-${symbol}-${entry.market}`, symbol, market: entry.market, note: entry.note, addedAt: Date.now() }, ...prev],
      );
      await supabase
        .from("stocker_watchlist")
        .upsert({ symbol, market: entry.market, note: entry.note ?? null }, { onConflict: "user_id,symbol,market", ignoreDuplicates: true });
      refresh();
    },
    [supabase, userId, refresh],
  );

  const remove = useCallback(
    async (id: string) => {
      setItems((prev) => prev.filter((p) => p.id !== id));
      await supabase.from("stocker_watchlist").delete().eq("id", id);
    },
    [supabase],
  );

  const has = useCallback(
    (symbol: string, market: Market) => items.some((p) => p.symbol === symbol.toUpperCase() && p.market === market),
    [items],
  );

  const clear = useCallback(async () => {
    setItems([]);
    if (userId) await supabase.from("stocker_watchlist").delete().eq("user_id", userId);
  }, [supabase, userId]);

  return { items, add, remove, has, clear, hydrated };
}

/* ----------------------------- paper trades ------------------------------ */

export function usePaperTrades() {
  const supabase = getSupabaseBrowser();
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [hydrated, setHydrated] = useState(false);

  const refresh = useCallback(async () => {
    if (!userId) {
      setTrades([]);
      setHydrated(true);
      return;
    }
    const { data } = await supabase
      .from("stocker_trades")
      .select("*")
      .order("opened_at", { ascending: false });
    setTrades((data ?? []).map(mapTradeRow));
    setHydrated(true);
  }, [supabase, userId]);

  useEffect(() => {
    setHydrated(false);
    refresh();
  }, [refresh]);

  const open = useCallback(
    async (t: Omit<PaperTrade, "id" | "openedAt" | "status">) => {
      if (!userId) return;
      setTrades((prev) => [{ ...t, id: `tmp-${Date.now()}`, openedAt: Date.now(), status: "open" }, ...prev]);
      await supabase.from("stocker_trades").insert({
        ticker: t.ticker,
        market: t.market,
        side: t.side,
        qty: t.qty,
        entry: t.entry,
        stop: t.stop,
        target: t.target,
        setup_family: t.setupFamily ?? null,
        score: t.score ?? null,
        grade: t.grade ?? null,
        pattern: t.pattern ?? null,
        source: t.source,
      });
      refresh();
    },
    [supabase, userId, refresh],
  );

  const close = useCallback(
    async (id: string, exit: number) => {
      setTrades((prev) =>
        prev.map((t) => (t.id === id ? { ...t, status: "closed", exit, closedAt: Date.now() } : t)),
      );
      await supabase
        .from("stocker_trades")
        .update({ status: "closed", exit, closed_at: new Date().toISOString() })
        .eq("id", id);
    },
    [supabase],
  );

  const update = useCallback(
    async (id: string, patch: Partial<PaperTrade>) => {
      setTrades((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
      const dbPatch: Row = {};
      if (patch.qty != null) dbPatch.qty = patch.qty;
      if (patch.entry != null) dbPatch.entry = patch.entry;
      if (patch.stop !== undefined) dbPatch.stop = patch.stop;
      if (patch.target !== undefined) dbPatch.target = patch.target;
      if (patch.note !== undefined) dbPatch.note = patch.note;
      if (Object.keys(dbPatch).length) await supabase.from("stocker_trades").update(dbPatch).eq("id", id);
    },
    [supabase],
  );

  const remove = useCallback(
    async (id: string) => {
      setTrades((prev) => prev.filter((t) => t.id !== id));
      await supabase.from("stocker_trades").delete().eq("id", id);
    },
    [supabase],
  );

  return { trades, open, close, update, remove, hydrated };
}

/* ------------------------------ computations ----------------------------- */

export interface TradeMetrics {
  mark: number | null;
  pnl: number | null;
  pnlPct: number | null;
  r: number | null;
}

/** Mark-to-market a trade. Closed trades use the exit; open ones the supplied live price. */
export function tradeMetrics(t: PaperTrade, livePrice: number | null): TradeMetrics {
  const mark = t.status === "closed" ? t.exit ?? null : livePrice;
  if (mark == null || !Number.isFinite(mark)) return { mark: null, pnl: null, pnlPct: null, r: null };
  const pnl = (mark - t.entry) * t.qty;
  const pnlPct = t.entry > 0 ? ((mark - t.entry) / t.entry) * 100 : null;
  const risk = t.stop != null ? t.entry - t.stop : null;
  const r = risk && risk > 0 ? (mark - t.entry) / risk : null;
  return { mark, pnl, pnlPct, r };
}

export interface BlotterSummary {
  openCount: number;
  closedCount: number;
  wins: number;
  losses: number;
  winRate: number | null;
  realizedPnl: number;
  unrealizedPnl: number;
  totalPnl: number;
  avgR: number | null;
  profitFactor: number | null;
  bestR: number | null;
  worstR: number | null;
}

/** Roll up the whole blotter. `prices` maps "TICKER|MARKET" -> live price for open trades. */
export function summarize(trades: PaperTrade[], prices: Record<string, number>): BlotterSummary {
  let wins = 0;
  let losses = 0;
  let realizedPnl = 0;
  let unrealizedPnl = 0;
  let grossWin = 0;
  let grossLoss = 0;
  const rs: number[] = [];
  let openCount = 0;
  let closedCount = 0;

  for (const t of trades) {
    const live = t.status === "open" ? prices[`${t.ticker}|${t.market}`] ?? null : null;
    const m = tradeMetrics(t, live);
    if (t.status === "closed") {
      closedCount += 1;
      if (m.pnl != null) {
        realizedPnl += m.pnl;
        if (m.pnl >= 0) {
          wins += 1;
          grossWin += m.pnl;
        } else {
          losses += 1;
          grossLoss += Math.abs(m.pnl);
        }
      }
      if (m.r != null) rs.push(m.r);
    } else {
      openCount += 1;
      if (m.pnl != null) unrealizedPnl += m.pnl;
    }
  }

  const decided = wins + losses;
  return {
    openCount,
    closedCount,
    wins,
    losses,
    winRate: decided > 0 ? (wins / decided) * 100 : null,
    realizedPnl,
    unrealizedPnl,
    totalPnl: realizedPnl + unrealizedPnl,
    avgR: rs.length ? rs.reduce((a, b) => a + b, 0) / rs.length : null,
    profitFactor: grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : null,
    bestR: rs.length ? Math.max(...rs) : null,
    worstR: rs.length ? Math.min(...rs) : null,
  };
}

/* ------------------------------ backup I/O ------------------------------- */

/** Export the signed-in user's data as a JSON string (for backup or device transfer). */
export async function exportData(): Promise<string> {
  const supabase = getSupabaseBrowser();
  const [w, t] = await Promise.all([
    supabase.from("stocker_watchlist").select("*"),
    supabase.from("stocker_trades").select("*"),
  ]);
  return JSON.stringify(
    { version: 2, exportedAt: Date.now(), watchlist: (w.data ?? []).map(mapWatchRow), trades: (t.data ?? []).map(mapTradeRow) },
    null,
    2,
  );
}

/** Bulk-import rows from an exported JSON string (also migrates old localStorage backups). */
export async function importData(json: string) {
  const supabase = getSupabaseBrowser();
  const data = JSON.parse(json) as { watchlist?: WatchItem[]; trades?: PaperTrade[] };

  if (Array.isArray(data.watchlist) && data.watchlist.length) {
    await supabase.from("stocker_watchlist").upsert(
      data.watchlist.map((w) => ({ symbol: String(w.symbol).toUpperCase(), market: w.market, note: w.note ?? null })),
      { onConflict: "user_id,symbol,market", ignoreDuplicates: true },
    );
  }
  if (Array.isArray(data.trades) && data.trades.length) {
    await supabase.from("stocker_trades").insert(
      data.trades.map((t) => ({
        ticker: t.ticker,
        market: t.market,
        side: t.side ?? "long",
        qty: t.qty,
        entry: t.entry,
        stop: t.stop ?? null,
        target: t.target ?? null,
        setup_family: t.setupFamily ?? null,
        score: t.score ?? null,
        grade: t.grade ?? null,
        pattern: t.pattern ?? null,
        source: t.source ?? "manual",
        status: t.status ?? "open",
        exit: t.exit ?? null,
        closed_at: t.closedAt ? new Date(t.closedAt).toISOString() : null,
        note: t.note ?? null,
      })),
    );
  }
}
