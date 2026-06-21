import "server-only";

export interface Headline {
  title: string;
  source: string;
  pubDate: string;
}

const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36";

const decode = (s: string) =>
  s
    .replace(/<!\[CDATA\[|\]\]>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(Number(d)))
    .trim();

/** Pull recent international headlines for a query from Google News RSS (free, no key). */
export async function fetchHeadlines(query: string, limit = 12): Promise<Headline[]> {
  const url = `https://news.google.com/rss/search?q=${encodeURIComponent(`${query} when:14d`)}&hl=en-US&gl=US&ceid=US:en`;
  const res = await fetch(url, { headers: { "User-Agent": UA }, next: { revalidate: 900 } });
  if (!res.ok) throw new Error(`News fetch failed (${res.status})`);
  const xml = await res.text();

  const items = [...xml.matchAll(/<item>([\s\S]*?)<\/item>/g)];
  const out: Headline[] = [];
  for (const m of items) {
    const block = m[1];
    const title = decode((block.match(/<title>([\s\S]*?)<\/title>/) || [])[1] ?? "");
    const source = decode((block.match(/<source[^>]*>([\s\S]*?)<\/source>/) || [])[1] ?? "");
    const pubDate = decode((block.match(/<pubDate>([\s\S]*?)<\/pubDate>/) || [])[1] ?? "");
    if (title) out.push({ title, source, pubDate });
    if (out.length >= limit) break;
  }
  return out;
}
