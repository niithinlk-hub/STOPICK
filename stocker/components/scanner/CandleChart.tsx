"use client";

import { useEffect, useRef } from "react";
import { CandlestickChart } from "lucide-react";
import type { Bar, SetupSignal } from "@/lib/engine/types";

/**
 * Dark-theme candlestick chart (lightweight-charts v4). Created entirely inside
 * an effect so the library never executes on the server. Draws horizontal price
 * lines for the breakout level, execution plan (entry/stop/targets) and any
 * retest / FVG zone bounds.
 */

function readToken(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (!raw) return fallback;
  // Tokens are space-separated channel triples ("52 211 153"). lightweight-charts'
  // color parser only accepts comma syntax, so normalize to rgb(r, g, b).
  return `rgb(${raw.split(/\s+/).join(", ")})`;
}

export function CandleChart({ bars, setup }: { bars: Bar[]; setup: SetupSignal }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || bars.length === 0 || typeof window === "undefined") return;

    let disposed = false;
    let cleanup: (() => void) | undefined;

    (async () => {
      const lc = await import("lightweight-charts");
      if (disposed || !containerRef.current) return;

      const bull = readToken("--bull", "rgb(52, 211, 153)");
      const bear = readToken("--bear", "rgb(248, 113, 113)");
      const text = readToken("--text", "rgb(230, 237, 243)");
      const brand = readToken("--brand", "rgb(34, 211, 238)");
      const info = readToken("--info", "rgb(56, 189, 248)");
      const grid = "rgba(148, 163, 184, 0.08)";
      const muted = "rgba(148, 163, 184, 0.55)";

      const chart = lc.createChart(containerRef.current, {
        height: 320,
        width: containerRef.current.clientWidth,
        layout: {
          background: { type: lc.ColorType.Solid, color: "transparent" },
          textColor: text,
          fontFamily: "var(--font-mono, monospace)",
        },
        grid: {
          vertLines: { color: grid },
          horzLines: { color: grid },
        },
        rightPriceScale: { borderColor: grid },
        timeScale: { borderColor: grid, timeVisible: true, secondsVisible: false },
        crosshair: { mode: lc.CrosshairMode.Normal },
        handleScale: true,
        handleScroll: true,
      });

      const series = chart.addCandlestickSeries({
        upColor: bull,
        downColor: bear,
        wickUpColor: bull,
        wickDownColor: bear,
        borderUpColor: bull,
        borderDownColor: bear,
      });

      series.setData(
        bars.map((b) => ({
          time: b.time as unknown as import("lightweight-charts").UTCTimestamp,
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
        })),
      );

      const line = (
        price: number | null | undefined,
        color: string,
        title: string,
        dashed = false,
      ) => {
        if (price === null || price === undefined || !Number.isFinite(price)) return;
        series.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle: dashed ? lc.LineStyle.Dashed : lc.LineStyle.Solid,
          axisLabelVisible: true,
          title,
        });
      };

      line(setup.breakout?.breakoutLevel ?? null, brand, "BO");

      const plan = setup.executionPlan;
      if (plan) {
        line(plan.entry, info, "Entry");
        line(plan.stop, bear, "Stop");
        line(plan.target1r, bull, "1R", true);
        line(plan.target2r, bull, "2R", true);
        line(plan.target3r, bull, "3R", true);
      }

      const zone = setup.structure?.retestZone ?? setup.structure?.fvgZone ?? null;
      if (zone) {
        line(zone[0], muted, "Zone↑");
        line(zone[1], muted, "Zone↓");
      }

      chart.timeScale().fitContent();

      const ro = new ResizeObserver((entries) => {
        const w = entries[0]?.contentRect.width;
        if (w) chart.applyOptions({ width: Math.floor(w) });
      });
      ro.observe(containerRef.current);

      cleanup = () => {
        ro.disconnect();
        chart.remove();
      };
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [bars, setup]);

  if (bars.length === 0) {
    return (
      <div className="flex h-[320px] flex-col items-center justify-center gap-2 text-center">
        <CandlestickChart className="h-8 w-8 text-faint" strokeWidth={1.75} />
        <p className="text-sm text-muted">No price data available for this symbol.</p>
      </div>
    );
  }

  return <div ref={containerRef} className="h-[320px] w-full" aria-label={`${setup.ticker} price chart`} />;
}
