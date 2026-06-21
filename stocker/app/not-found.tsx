import Link from "next/link";
import { Compass } from "lucide-react";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-4 px-4 py-24 text-center">
      <span className="grid h-12 w-12 place-items-center rounded-xl bg-brand/10 text-brand ring-1 ring-brand/30">
        <Compass className="h-6 w-6" />
      </span>
      <h1 className="text-lg font-semibold tracking-tight">Page not found</h1>
      <p className="text-sm text-muted">That route does not exist in STOCKER.</p>
      <Link
        href="/"
        className="rounded-lg border border-border bg-elevated px-3.5 py-2 text-sm text-text transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
      >
        Back to Scanner
      </Link>
    </div>
  );
}
