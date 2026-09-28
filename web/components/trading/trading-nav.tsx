"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FlaskConical } from "lucide-react";
import { cn } from "@/lib/utils";

const TABS = [
  { href: "/trading", label: "Workspace" },
  { href: "/trading/strategies", label: "Strategies" },
  { href: "/trading/dispatches", label: "Dispatch history" },
] as const;

/** Sub-navigation shared by the trading pages. */
export function TradingNav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Trading sections" className="flex flex-wrap gap-1 border-b">
      {TABS.map((t) => {
        const active = pathname === t.href;
        return (
          <Link
            key={t.href}
            href={t.href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "-mb-px rounded-t-md border-b-2 px-3 py-2 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              active
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}

/**
 * Loud, persistent label that the trading venue is simulated. Shows the
 * venue name reported by the backend so a future real venue is visible too.
 */
export function VenueBadge({ venue }: { venue: string | undefined }) {
  const simulated = !venue || venue.toLowerCase() === "simulated";
  return (
    <span
      data-testid="venue-badge"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-semibold uppercase tracking-wide",
        simulated
          ? "border-amber-400 bg-amber-50 text-amber-900 dark:bg-amber-950 dark:text-amber-200"
          : "border-emerald-400 bg-emerald-50 text-emerald-900",
      )}
      title={
        simulated
          ? "Orders are matched by an in-process simulated venue. Nothing is sent to a real market."
          : `Venue: ${venue}`
      }
    >
      {simulated && <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />}
      {simulated ? "Simulated venue" : `Venue: ${venue}`}
    </span>
  );
}
