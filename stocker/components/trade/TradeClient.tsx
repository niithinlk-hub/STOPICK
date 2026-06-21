"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { AlertTriangle, ExternalLink, Loader2, RefreshCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";
import { useAuth } from "@/components/auth/AuthProvider";
import { SignInNotice } from "@/components/auth/SignInNotice";

const inr = (n: number | null | undefined) =>
  n == null || Number.isNaN(n) ? "—" : `₹${Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;

interface Funds {
  availabelBalance?: number;
  withdrawableBalance?: number;
  utilizedAmount?: number;
  collateralAmount?: number;
  sodLimit?: number;
}
interface Position {
  tradingSymbol?: string;
  netQty?: number;
  buyAvg?: number;
  costPrice?: number;
  realizedProfit?: number;
  unrealizedProfit?: number;
  productType?: string;
}
interface Holding {
  tradingSymbol?: string;
  totalQty?: number;
  availableQty?: number;
  avgCostPrice?: number;
  lastTradedPrice?: number;
}
interface OrderRow {
  orderId?: string;
  tradingSymbol?: string;
  transactionType?: string;
  orderType?: string;
  orderStatus?: string;
  quantity?: number;
  price?: number;
  productType?: string;
}
interface Portfolio {
  token?: { expired?: boolean; hoursLeft?: number | null; autoRefresh?: boolean };
  funds?: Funds | null;
  positions?: Position[];
  holdings?: Holding[];
  orders?: OrderRow[];
  error?: string;
}

const pnlClass = (n: number | null | undefined) =>
  n == null ? "text-muted" : n > 0 ? "text-bull" : n < 0 ? "text-bear" : "text-text";

export function TradeClient() {
  const { user, loading: authLoading } = useAuth();
  const [data, setData] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [forbidden, setForbidden] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetch("/api/dhan/portfolio", { cache: "no-store" });
      if (r.status === 403) {
        setForbidden(true);
        setData(null);
        return;
      }
      const j = (await r.json()) as Portfolio;
      setData(j);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load portfolio.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user) load();
    else if (!authLoading) setLoading(false);
  }, [user, authLoading, load]);

  if (!authLoading && !user) {
    return (
      <div className="mx-auto max-w-[960px] px-4 py-5 sm:px-6">
        <SignInNotice feature="your Dhan portfolio" />
      </div>
    );
  }

  const positions = data?.positions ?? [];
  const holdings = data?.holdings ?? [];
  const orders = data?.orders ?? [];
  const openOrders = orders.filter((o) => !["TRADED", "CANCELLED", "REJECTED", "EXPIRED"].includes(o.orderStatus ?? ""));
  const f = data?.funds ?? null;
  const posPnl = positions.reduce((s, p) => s + (p.unrealizedProfit ?? 0) + (p.realizedProfit ?? 0), 0);
  const holdingsPnl = holdings.reduce(
    (s, h) => s + (h.lastTradedPrice ? (h.lastTradedPrice - (h.avgCostPrice ?? 0)) * (h.totalQty ?? 0) : 0),
    0,
  );

  const th = "px-3 py-2 text-left text-2xs font-medium uppercase tracking-widest text-faint";
  const td = "px-3 py-2 text-sm text-text";

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-5 sm:px-6">
      <header className="mb-5 flex items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Trade — Dhan</h1>
          <p className="mt-0.5 text-sm text-muted">
            Live positions, holdings, orders and funds from your Dhan account (read-only). Place orders from{" "}
            <span className="font-medium text-text">Trade on Dhan</span> on any setup.
          </p>
        </div>
        <Button type="button" variant="secondary" onClick={load} disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          Refresh
        </Button>
      </header>

      {forbidden && (
        <Card>
          <CardBody className="flex items-center gap-2 py-6 text-sm text-muted">
            <AlertTriangle className="h-4 w-4 text-warn" />
            The Dhan account is owner-only. Sign in as the admin to view the portfolio.
          </CardBody>
        </Card>
      )}

      {err && <p className="mb-4 text-sm text-bear">{err}</p>}

      {data?.error && (
        <Card>
          <CardBody className="flex flex-wrap items-center gap-2 py-6 text-sm text-muted">
            <AlertTriangle className="h-4 w-4 text-warn" />
            {data.error}{" "}
            <Link href="/admin" className="font-medium text-brand hover:underline">
              Open Settings →
            </Link>
          </CardBody>
        </Card>
      )}

      {data && !data.error && (
        <div className="space-y-5">
          {data.token?.expired && (
            <p className="rounded-lg border border-bear/30 bg-bear/10 px-3 py-2 text-xs text-bear">
              Dhan token expired — portfolio may be stale.{" "}
              {data.token.autoRefresh ? "Auto-refresh will mint a new one on the next scan." : "Refresh it in Settings."}
            </p>
          )}

          {/* Funds */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: "Available", value: f?.availabelBalance },
              { label: "Withdrawable", value: f?.withdrawableBalance },
              { label: "Utilized", value: f?.utilizedAmount },
              { label: "Collateral", value: f?.collateralAmount },
            ].map((x) => (
              <Card key={x.label}>
                <CardBody className="py-3">
                  <p className="text-2xs font-medium uppercase tracking-widest text-faint">{x.label}</p>
                  <p className="mt-1 text-base font-semibold tabular-nums text-text">{inr(x.value)}</p>
                </CardBody>
              </Card>
            ))}
          </div>

          {/* Positions */}
          <Card>
            <CardHeader className="flex items-center justify-between">
              <CardTitle>Positions</CardTitle>
              {positions.length > 0 && (
                <span className={cn("text-sm font-semibold tabular-nums", pnlClass(posPnl))}>{inr(posPnl)} P&L</span>
              )}
            </CardHeader>
            <CardBody className="p-0">
              {positions.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-muted">No open positions.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        <th className={th}>Symbol</th>
                        <th className={th}>Qty</th>
                        <th className={th}>Avg</th>
                        <th className={th}>Product</th>
                        <th className={cn(th, "text-right")}>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((p, i) => {
                        const pnl = (p.unrealizedProfit ?? 0) + (p.realizedProfit ?? 0);
                        return (
                          <tr key={`${p.tradingSymbol}-${i}`} className="border-b border-border/60 last:border-0">
                            <td className={cn(td, "font-medium")}>{p.tradingSymbol ?? "—"}</td>
                            <td className={cn(td, "tabular-nums")}>{p.netQty ?? 0}</td>
                            <td className={cn(td, "tabular-nums")}>{inr(p.buyAvg ?? p.costPrice)}</td>
                            <td className={td}>
                              <Badge>{p.productType ?? "—"}</Badge>
                            </td>
                            <td className={cn(td, "text-right font-semibold tabular-nums", pnlClass(pnl))}>{inr(pnl)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardBody>
          </Card>

          {/* Holdings */}
          <Card>
            <CardHeader className="flex items-center justify-between">
              <CardTitle>Holdings</CardTitle>
              {holdings.length > 0 && (
                <span className={cn("text-sm font-semibold tabular-nums", pnlClass(holdingsPnl))}>{inr(holdingsPnl)} P&L</span>
              )}
            </CardHeader>
            <CardBody className="p-0">
              {holdings.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-muted">No holdings.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        <th className={th}>Symbol</th>
                        <th className={th}>Qty</th>
                        <th className={th}>Avg</th>
                        <th className={th}>LTP</th>
                        <th className={cn(th, "text-right")}>Value</th>
                        <th className={cn(th, "text-right")}>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {holdings.map((h, i) => {
                        const qty = h.totalQty ?? 0;
                        const ltp = h.lastTradedPrice ?? 0;
                        const pnl = ltp ? (ltp - (h.avgCostPrice ?? 0)) * qty : null;
                        return (
                          <tr key={`${h.tradingSymbol}-${i}`} className="border-b border-border/60 last:border-0">
                            <td className={cn(td, "font-medium")}>{h.tradingSymbol ?? "—"}</td>
                            <td className={cn(td, "tabular-nums")}>{qty}</td>
                            <td className={cn(td, "tabular-nums")}>{inr(h.avgCostPrice)}</td>
                            <td className={cn(td, "tabular-nums")}>{inr(h.lastTradedPrice)}</td>
                            <td className={cn(td, "text-right tabular-nums")}>{inr(ltp * qty)}</td>
                            <td className={cn(td, "text-right font-semibold tabular-nums", pnlClass(pnl))}>{inr(pnl)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardBody>
          </Card>

          {/* Open orders */}
          <Card>
            <CardHeader>
              <CardTitle>Open orders</CardTitle>
            </CardHeader>
            <CardBody className="p-0">
              {openOrders.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-muted">No open orders.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        <th className={th}>Symbol</th>
                        <th className={th}>Side</th>
                        <th className={th}>Type</th>
                        <th className={th}>Qty</th>
                        <th className={th}>Price</th>
                        <th className={th}>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {openOrders.map((o, i) => (
                        <tr key={o.orderId ?? i} className="border-b border-border/60 last:border-0">
                          <td className={cn(td, "font-medium")}>{o.tradingSymbol ?? "—"}</td>
                          <td className={cn(td, o.transactionType === "SELL" ? "text-bear" : "text-bull")}>
                            {o.transactionType ?? "—"}
                          </td>
                          <td className={td}>{o.orderType ?? "—"}</td>
                          <td className={cn(td, "tabular-nums")}>{o.quantity ?? 0}</td>
                          <td className={cn(td, "tabular-nums")}>{inr(o.price)}</td>
                          <td className={td}>
                            <Badge>{o.orderStatus ?? "—"}</Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardBody>
          </Card>

          <a
            href="https://web.dhan.co/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-brand hover:underline"
          >
            Open full Dhan trader <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      )}
    </div>
  );
}
