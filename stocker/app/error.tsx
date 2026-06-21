"use client";

import { useEffect } from "react";
import { AlertTriangle, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/primitives";

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.error(error);
  }, [error]);

  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-4 px-4 py-24 text-center">
      <span className="grid h-12 w-12 place-items-center rounded-xl bg-bear/10 text-bear ring-1 ring-bear/30">
        <AlertTriangle className="h-6 w-6" />
      </span>
      <h1 className="text-lg font-semibold tracking-tight">Something went wrong</h1>
      <p className="text-sm text-muted">{error.message || "An unexpected error occurred while rendering this view."}</p>
      <Button onClick={reset} variant="secondary">
        <RotateCw className="h-4 w-4" /> Try again
      </Button>
    </div>
  );
}
