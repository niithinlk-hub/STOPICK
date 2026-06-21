"use client";

import { useState } from "react";
import { Check, Star, Wallet } from "lucide-react";
import { Button } from "@/components/ui/primitives";
import { useWatchlist, usePaperTrades } from "@/lib/client/store";
import type { SetupSignal } from "@/lib/engine/types";

/**
 * One-click "Add to watchlist" / "Paper trade this" actions for a scanned setup.
 * The paper trade is pre-filled from the execution plan (entry / stop / 2R target /
 * position size), so the blotter mirrors exactly what the scanner proposed.
 */
export function SetupActions({ setup }: { setup: SetupSignal }) {
  const watchlist = useWatchlist();
  const trades = usePaperTrades();
  const [logged, setLogged] = useState(false);

  const plan = setup.executionPlan;
  const onWatch = watchlist.has(setup.ticker, setup.market);

  const logTrade = () => {
    if (!plan) return;
    trades.open({
      ticker: setup.ticker,
      market: setup.market,
      side: "long",
      qty: Math.max(0, Math.round(plan.positionSizeShares)),
      entry: plan.entry,
      stop: plan.stop,
      target: plan.target2r,
      setupFamily: setup.setupFamily,
      score: setup.score,
      grade: setup.grade,
      pattern: setup.breakout?.patternName,
      source: "scan",
    });
    setLogged(true);
    window.setTimeout(() => setLogged(false), 2200);
  };

  return (
    <div className="flex flex-wrap gap-2">
      <Button
        variant={onWatch ? "secondary" : "primary"}
        onClick={() => !onWatch && watchlist.add({ symbol: setup.ticker, market: setup.market })}
        disabled={onWatch}
      >
        {onWatch ? <Check className="h-4 w-4" strokeWidth={2} /> : <Star className="h-4 w-4" strokeWidth={2} />}
        {onWatch ? "On watchlist" : "Add to watchlist"}
      </Button>
      <Button variant="secondary" onClick={logTrade} disabled={!plan}>
        {logged ? <Check className="h-4 w-4 text-bull" strokeWidth={2} /> : <Wallet className="h-4 w-4" strokeWidth={2} />}
        {logged ? "Logged to blotter" : "Paper trade this"}
      </Button>
    </div>
  );
}
