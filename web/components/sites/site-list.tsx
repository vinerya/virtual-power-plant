"use client";

import { useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import type { Site } from "@/lib/api/types";
import { cn, formatPower } from "@/lib/utils";

const HEALTH_DOT = {
  green: "bg-emerald-500",
  yellow: "bg-amber-500",
  red: "bg-destructive",
} as const;

export function SiteList({
  sites,
  selectedId,
  onSelect,
}: {
  sites: Site[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return sites;
    return sites.filter(
      (s) =>
        s.name.toLowerCase().includes(needle) ||
        s.id.toLowerCase().includes(needle),
    );
  }, [sites, q]);

  return (
    <aside
      aria-label="Sites"
      className="flex h-full w-72 flex-shrink-0 flex-col border-r"
      data-testid="site-list"
    >
      <div className="border-b p-3">
        <div className="relative">
          <Search
            aria-hidden="true"
            className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            aria-label="Search sites"
            placeholder="Search sites…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="pl-7"
          />
        </div>
      </div>
      <ul className="flex-1 overflow-auto" role="listbox" aria-label="Sites">
        {filtered.length === 0 ? (
          <li className="p-3 text-sm text-muted-foreground">No sites match.</li>
        ) : (
          filtered.map((s) => {
            const active = selectedId === s.id;
            return (
              <li key={s.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={active}
                  onClick={() => onSelect(s.id)}
                  data-testid="site-row"
                  data-site-id={s.id}
                  className={cn(
                    "flex w-full items-center gap-3 border-b px-3 py-2 text-left text-sm hover:bg-muted/40 focus-visible:bg-muted/50 focus-visible:outline-none",
                    active && "bg-primary/10 text-primary",
                  )}
                >
                  <span
                    aria-hidden="true"
                    className={cn(
                      "h-2.5 w-2.5 flex-shrink-0 rounded-full",
                      HEALTH_DOT[s.health],
                    )}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="line-clamp-1 font-medium">{s.name}</span>
                    <span className="block text-xs text-muted-foreground">
                      {s.online_count}/{s.total_resources} online ·{" "}
                      {formatPower(s.current_power)}
                    </span>
                  </span>
                  {s.active_alerts > 0 && (
                    <span className="rounded-full bg-destructive/10 px-1.5 py-0.5 text-[10px] font-medium text-destructive">
                      {s.active_alerts}
                    </span>
                  )}
                </button>
              </li>
            );
          })
        )}
      </ul>
    </aside>
  );
}
