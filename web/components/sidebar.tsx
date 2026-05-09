"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Cog,
  Layers,
  LineChart,
  Receipt,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Fleet", icon: BarChart3, enabled: true },
  { href: "/assets", label: "Assets", icon: Layers, enabled: true },
  { href: "/trading/dispatches", label: "Dispatches", icon: LineChart, enabled: true },
  { href: "/tariffs", label: "Tariffs", icon: Receipt, enabled: false },
  { href: "/alerts", label: "Alerts", icon: AlertTriangle, enabled: false },
  { href: "/settings", label: "Settings", icon: Cog, enabled: false },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden w-56 flex-shrink-0 border-r bg-card md:block">
      <div className="flex h-14 items-center border-b px-4">
        <Activity className="mr-2 h-5 w-5 text-primary" aria-hidden="true" />
        <span className="text-sm font-semibold">VPP Console</span>
      </div>
      <nav aria-label="Primary" className="space-y-1 p-2">
        {NAV.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
                !item.enabled && "opacity-70",
              )}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              <span>{item.label}</span>
              {!item.enabled && (
                <span className="ml-auto text-[10px] uppercase tracking-wider text-muted-foreground">
                  soon
                </span>
              )}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
