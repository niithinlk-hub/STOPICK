"use client";

import { useState } from "react";
import { Check, Plus } from "lucide-react";
import { cn } from "@/lib/cn";
import { usePaperTrades } from "@/lib/client/store";
import type { Grade, Market, SetupFamily } from "@/lib/engine/types";

/** 0.5% of the 1,000,000 default book — matches the engine's position sizing. */
const RISK_BUDGET = 5000;

const fmtPrice = (v: number | null) =>
  v == null || !Number.isFinite(v) ? "" : (Math.round(v * 100) / 100).toString();

const suggestQty = (price: number, stop: number | null) =>
  stop != null && price > stop ? Math.max(1, Math.floor(RISK_BUDGET / (price - stop))) : 1;

/**
 * Inline "add to paper trade" control for a screener row: an editable entry-PRICE
 * box + quantity box + one-click log. The price defaults to the row's suggested
 * entry level, but the user sets the price they actually entered at; quantity is
 * auto-sized from that price vs the stop so risk lands near the engine default —
 * until the user overrides the quantity, after which it's left alone.
 */
export function QuickAddTrade(props: {
  ticker: string;
  market: Market;
  entry: number | null;
  stop: number | null;
  target: number | null;
  setupFamily?: SetupFamily;
  score?: number;
  grade?: Grade;
  pattern?: string;
  source: "scan" | "manual";
}) {
  const { open } = usePaperTrades();
  const { stop, target } = props;

  const [price, setPrice] = useState(fmtPrice(props.entry));
  const [qty, setQty] = useState(String(suggestQty(props.entry ?? 0, stop)));
  const [qtyDirty, setQtyDirty] = useState(false);
  const [added, setAdded] = useState(false);

  const priceNum = Number(price);
  const qtyNum = Number(qty);
  const priceValid = price.trim() !== "" && Number.isFinite(priceNum) && priceNum > 0;
  const valid = priceValid && Number.isFinite(qtyNum) && qtyNum > 0;

  const onPriceChange = (raw: string) => {
    // digits + a single decimal point
    const cleaned = raw.replace(/[^0-9.]/g, "").replace(/(\..*)\./g, "$1");
    setPrice(cleaned);
    const p = Number(cleaned);
    if (!qtyDirty && Number.isFinite(p) && p > 0) setQty(String(suggestQty(p, stop)));
  };

  const add = () => {
    if (!valid) return;
    open({
      ticker: props.ticker,
      market: props.market,
      side: "long",
      qty: Math.round(qtyNum),
      entry: priceNum,
      stop,
      target,
      setupFamily: props.setupFamily,
      score: props.score,
      grade: props.grade,
      pattern: props.pattern,
      source: props.source,
    });
    setAdded(true);
    window.setTimeout(() => setAdded(false), 1500);
  };

  const box =
    "rounded-md border border-border bg-elevated/60 px-1.5 py-1 text-right font-mono text-2xs tnum text-text placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";

  return (
    <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
      <input
        value={price}
        onChange={(e) => onPriceChange(e.target.value)}
        inputMode="decimal"
        placeholder="price"
        aria-label={`Entry price for ${props.ticker}`}
        title="Entry price you bought at"
        className={cn(box, "w-16")}
      />
      <input
        value={qty}
        onChange={(e) => {
          setQty(e.target.value.replace(/[^0-9]/g, ""));
          setQtyDirty(true);
        }}
        inputMode="numeric"
        aria-label={`Quantity for ${props.ticker}`}
        title="Quantity"
        className={cn(box, "w-14")}
      />
      <button
        type="button"
        onClick={add}
        disabled={!valid}
        aria-label={`Add ${props.ticker} to paper trades`}
        title={!priceValid ? "Enter an entry price" : "Add to paper trades"}
        className={cn(
          "rounded-md p-1 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
          "text-muted hover:bg-brand/10 hover:text-brand disabled:cursor-not-allowed disabled:opacity-40",
        )}
      >
        {added ? <Check className="h-3.5 w-3.5 text-bull" strokeWidth={2.5} /> : <Plus className="h-3.5 w-3.5" strokeWidth={2.5} />}
      </button>
    </div>
  );
}
