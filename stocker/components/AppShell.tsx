"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  BarChart3,
  Briefcase,
  Coins,
  FlaskConical,
  Gauge,
  GitMerge,
  LineChart,
  LogIn,
  LogOut,
  Menu,
  Radar,
  Settings,
  Star,
  TrendingUp,
  Wallet,
  X,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { useAuth } from "@/components/auth/AuthProvider";
import { ThemeToggle } from "@/components/theme/ThemeToggle";
import { StockerLogo } from "@/components/brand/StockerLogo";
import { InstallButton } from "@/components/pwa/InstallButton";

const NAV = [
  { href: "/", label: "Scanner", icon: Radar },
  { href: "/coiling", label: "Coiling", icon: Zap },
  { href: "/top-setups", label: "Top Setups", icon: TrendingUp },
  { href: "/relative-strength", label: "Relative Strength", icon: LineChart },
  { href: "/regime", label: "Market Regime", icon: Gauge },
  { href: "/commodities", label: "Commodities", icon: Coins },
  { href: "/confluence", label: "RSI Confluence", icon: GitMerge },
  { href: "/backtest", label: "Backtest", icon: BarChart3 },
  { href: "/algo-check", label: "Algo Check", icon: FlaskConical },
  { href: "/watchlist", label: "Watchlist", icon: Star },
  { href: "/paper", label: "Paper Trades", icon: Wallet },
  { href: "/trade", label: "Trade", icon: Briefcase },
  { href: "/admin", label: "Settings", icon: Settings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  // Auth pages render standalone (no nav chrome).
  if (pathname === "/login" || pathname.startsWith("/reset-password")) {
    return <div className="app-grid-bg min-h-dvh">{children}</div>;
  }

  const nav = (
    <nav className="flex flex-col gap-1" aria-label="Primary">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            onClick={() => setOpen(false)}
            aria-current={active ? "page" : undefined}
            className={cn(
              "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60",
              active
                ? "bg-brand/10 text-text ring-1 ring-brand/30"
                : "text-muted hover:bg-white/[0.04] hover:text-text",
            )}
          >
            <Icon className={cn("h-[18px] w-[18px]", active ? "text-brand" : "text-faint group-hover:text-muted")} strokeWidth={2} />
            {label}
          </Link>
        );
      })}
    </nav>
  );

  return (
    <div className="app-grid-bg flex min-h-dvh">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col border-r border-border bg-surface/60 px-4 py-5 backdrop-blur lg:flex">
        <div className="flex items-center justify-between gap-2">
          <Brand />
          <ThemeToggle />
        </div>
        <div className="mt-7">{nav}</div>
        <div className="mt-auto space-y-3 pt-6">
          <InstallButton />
          <AccountBox />
          <DataSourceFooter />
        </div>
      </aside>

      {/* Mobile top bar */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-40 flex items-center justify-between border-b border-border bg-bg/80 px-4 py-3 backdrop-blur lg:hidden">
          <Brand compact />
          <div className="flex items-center gap-1">
            <ThemeToggle />
            <button
              type="button"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen((v) => !v)}
              className="rounded-lg p-2 text-muted hover:bg-overlay hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
            >
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </header>

        {open && (
          <div className="border-b border-border bg-surface px-4 py-3 lg:hidden">{nav}</div>
        )}

        <main className="min-w-0 flex-1">{children}</main>
      </div>
    </div>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <StockerLogo size={compact ? 30 : 38} />
      {!compact ? (
        <div className="leading-tight">
          <div className="text-base font-bold tracking-tight text-text">STOCKER</div>
          <div className="text-2xs uppercase tracking-[0.22em] text-faint">Analytics</div>
        </div>
      ) : (
        <span className="text-base font-bold tracking-tight">STOCKER</span>
      )}
    </div>
  );
}

function AccountBox() {
  const { user, loading, signOut } = useAuth();
  if (loading) return <div className="h-9 animate-pulse rounded-lg bg-elevated/60" />;
  if (!user) {
    return (
      <Link
        href="/login"
        className="flex items-center gap-2 rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm font-medium text-text transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
      >
        <LogIn className="h-4 w-4 text-brand" strokeWidth={2} />
        Sign in
      </Link>
    );
  }
  return (
    <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-elevated/60 px-3 py-2">
      <span className="min-w-0 truncate text-xs text-muted" title={user.email ?? undefined}>
        {user.email}
      </span>
      <button
        type="button"
        onClick={() => signOut()}
        aria-label="Sign out"
        className="rounded-md p-1.5 text-faint transition-colors hover:bg-white/[0.05] hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60"
      >
        <LogOut className="h-4 w-4" strokeWidth={2} />
      </button>
    </div>
  );
}

function DataSourceFooter() {
  return (
    <div className="rounded-lg border border-border bg-elevated/60 p-3 text-2xs leading-relaxed text-faint">
      Data via Yahoo Finance. Signals are deterministic and rule-based — confidence is not a
      win-rate claim. For research, not investment advice.
    </div>
  );
}
