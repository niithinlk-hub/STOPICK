/**
 * Process-level TTL cache + a small concurrency limiter. Mirrors STOPICK's disk
 * cache intent (avoid refetching OHLCV during a scan) but in-memory, which is the
 * right fit for a Next.js server runtime.
 */
const TTL_MS = 30 * 60 * 1000; // 30 minutes, matching STOPICK cache_ttl_minutes.

interface Entry<T> {
  value: T;
  expires: number;
}

const store = new Map<string, Entry<unknown>>();

export function cacheGet<T>(key: string): T | undefined {
  const entry = store.get(key);
  if (!entry) return undefined;
  if (Date.now() > entry.expires) {
    store.delete(key);
    return undefined;
  }
  return entry.value as T;
}

export function cacheSet<T>(key: string, value: T, ttlMs = TTL_MS): void {
  store.set(key, { value, expires: Date.now() + ttlMs });
}

/** Run async tasks with a bounded concurrency (default 8, like STOPICK max_workers). */
export async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const i = cursor++;
      results[i] = await worker(items[i], i);
    }
  });
  await Promise.all(runners);
  return results;
}
