"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getQuote } from "./api";
import type { Market } from "@/lib/engine/types";

export interface QuoteState {
  price: number;
  prevClose: number;
  asOf: number;
}

const keyOf = (t: { ticker: string; market: Market }) => `${t.ticker}|${t.market}`;

/**
 * Fetch last prices for a set of symbols and keep them keyed by "TICKER|MARKET".
 * Re-fetches when the target set changes; `refresh()` re-pulls on demand.
 * Failed symbols are simply omitted from the map (caller renders "—").
 */
export function useQuotes(targets: { ticker: string; market: Market }[]) {
  const [quotes, setQuotes] = useState<Record<string, QuoteState>>({});
  const [loading, setLoading] = useState(false);
  const [refreshedAt, setRefreshedAt] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Stable signature so the effect only re-runs when the symbol set changes.
  const sig = targets.map(keyOf).sort().join(",");

  const load = useCallback(async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const list = sig
      ? sig.split(",").map((s) => {
          const [ticker, market] = s.split("|");
          return { ticker, market: market as Market };
        })
      : [];
    if (!list.length) {
      setQuotes({});
      setLoading(false);
      return;
    }

    setLoading(true);
    const results = await Promise.allSettled(list.map((t) => getQuote(t.ticker, t.market, ctrl.signal)));
    if (ctrl.signal.aborted) return;
    const next: Record<string, QuoteState> = {};
    results.forEach((r, i) => {
      if (r.status === "fulfilled") {
        next[keyOf(list[i])] = { price: r.value.price, prevClose: r.value.prevClose, asOf: r.value.asOf };
      }
    });
    setQuotes(next);
    setLoading(false);
    setRefreshedAt(Date.now());
  }, [sig]);

  useEffect(() => {
    load();
    return () => abortRef.current?.abort();
  }, [load]);

  return { quotes, loading, refreshedAt, refresh: load };
}
