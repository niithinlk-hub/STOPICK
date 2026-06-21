import { NextResponse } from "next/server";
import { fetchOhlcv } from "@/lib/data/yahoo";
import { fetchHeadlines } from "@/lib/data/news";
import { analyzeNews } from "@/lib/server/llm";
import { getUserSettings } from "@/lib/server/settings";
import {
  COMMODITIES,
  predictCommodity,
  type CommodityNewsBlock,
  type CommodityPrediction,
  type NewsSentiment,
} from "@/lib/engine/commodities";
import { closes, last } from "@/lib/engine/indicators";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET() {
  const settings = await getUserSettings();
  const newsConfigured = Boolean(
    settings?.newsEnabled && settings.llmApiKey && settings.llmModel && settings.llmBaseUrl,
  );
  const notes: string[] = [];
  if (!newsConfigured) {
    notes.push("Predictions are technical-only. Add an AI model key in Settings to fuse live news sentiment.");
  }

  const commodities: CommodityPrediction[] = await Promise.all(
    COMMODITIES.map(async (def): Promise<CommodityPrediction> => {
      const bars = await fetchOhlcv(def.symbol, "1d", 300);
      const c = closes(bars);
      const price = last(c);
      const prev = c.length > 1 ? c[c.length - 2] : price;
      const dayChangePct = prev ? ((price - prev) / prev) * 100 : null;
      const asOf = (bars.length ? bars[bars.length - 1].time : Math.floor(Date.now() / 1000)) * 1000;

      let newsSent: NewsSentiment | null = null;
      let newsBlock: CommodityNewsBlock | null = null;

      if (newsConfigured) {
        try {
          const headlines = await fetchHeadlines(def.newsQuery, 12);
          const analysis = await analyzeNews(def.name, headlines, {
            baseUrl: settings!.llmBaseUrl!,
            model: settings!.llmModel!,
            apiKey: settings!.llmApiKey!,
          });
          const top = headlines.slice(0, 6).map((h) => ({ title: h.title, source: h.source }));
          newsSent = { ...analysis, headlines: top };
          newsBlock = { ...analysis, headlines: top, enabled: true };
        } catch (err) {
          newsBlock = {
            score: 0,
            label: "Unavailable",
            summary: "",
            drivers: [],
            headlines: [],
            enabled: true,
            error: err instanceof Error ? err.message : "News analysis failed.",
          };
        }
      }

      return {
        key: def.key,
        name: def.name,
        symbol: def.symbol,
        unit: def.unit,
        price,
        dayChangePct,
        horizons: predictCommodity(bars, newsSent),
        news: newsBlock,
        asOf,
      };
    }),
  );

  return NextResponse.json({ commodities, generatedAt: Date.now(), newsConfigured, notes });
}
