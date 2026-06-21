"use client";

import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { cn } from "@/lib/cn";

// Chrome's non-standard install-prompt event. Typed locally since it's not in lib.dom.
type BeforeInstallPromptEvent = Event & {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
};

/**
 * "Install app" affordance. Appears only when the browser fires
 * `beforeinstallprompt` and the app isn't already running standalone.
 */
export function InstallButton({ className }: { className?: string }) {
  const [deferred, setDeferred] = useState<BeforeInstallPromptEvent | null>(null);

  useEffect(() => {
    // Already installed / running standalone — never show the button.
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(display-mode: standalone)").matches
    ) {
      return;
    }

    const onPrompt = (e: Event) => {
      e.preventDefault();
      setDeferred(e as BeforeInstallPromptEvent);
    };
    const onInstalled = () => setDeferred(null);

    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  if (!deferred) return null;

  const onClick = async () => {
    try {
      await deferred.prompt();
      await deferred.userChoice;
    } catch {
      /* user dismissed or the prompt is no longer valid */
    } finally {
      setDeferred(null);
    }
  };

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm font-medium text-text transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
        className,
      )}
    >
      <Download className="h-4 w-4 text-brand" strokeWidth={2} />
      Install app
    </button>
  );
}
