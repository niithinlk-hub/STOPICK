"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";
import { cn } from "@/lib/cn";

type Theme = "light" | "dark";

/** Inline script (runs before paint, in layout) to apply the saved/system theme with no flash. */
export const themeNoFlashScript = `(function(){try{var t=localStorage.getItem('stocker-theme');if(t!=='light'&&t!=='dark'){t=window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';}document.documentElement.classList.toggle('light',t==='light');}catch(e){}})();`;

function resolveTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  const stored = localStorage.getItem("stocker-theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setTheme(resolveTheme());
    setMounted(true);
  }, []);

  const toggle = () => {
    const next: Theme = theme === "light" ? "dark" : "light";
    setTheme(next);
    try {
      localStorage.setItem("stocker-theme", next);
    } catch {
      /* storage disabled — still apply for this session */
    }
    document.documentElement.classList.toggle("light", next === "light");
  };

  const isLight = theme === "light";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isLight ? "Switch to dark theme" : "Switch to light theme"}
      title={isLight ? "Dark mode" : "Light mode"}
      className={cn(
        "rounded-lg p-2 text-muted transition-colors hover:bg-overlay hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
        className,
      )}
    >
      {/* Until mounted, render the moon (matches SSR dark default) to avoid a hydration flip. */}
      {mounted && isLight ? <Moon className="h-[18px] w-[18px]" strokeWidth={2} /> : <Sun className="h-[18px] w-[18px]" strokeWidth={2} />}
    </button>
  );
}
