"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Plus, Search } from "lucide-react";
import { listTariffs } from "@/lib/api/tariffs";
import { cn } from "@/lib/utils";

export function TariffList({
  selectedId,
  onSelect,
  onNew,
}: {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNew?: () => void;
}) {
  const [q, setQ] = useState("");
  const [utility, setUtility] = useState<string>("");

  const tariffs = useQuery({
    queryKey: ["tariffs"],
    queryFn: listTariffs,
    staleTime: 60_000,
  });

  const utilities = useMemo(() => {
    const set = new Set<string>();
    (tariffs.data ?? []).forEach((t) => t.utility && set.add(t.utility));
    return Array.from(set).sort();
  }, [tariffs.data]);

  const filtered = useMemo(() => {
    return (tariffs.data ?? []).filter((t) => {
      if (utility && t.utility !== utility) return false;
      if (q && !`${t.name} ${t.utility ?? ""}`.toLowerCase().includes(q.toLowerCase()))
        return false;
      return true;
    });
  }, [tariffs.data, q, utility]);

  return (
    <aside
      aria-label="Tariffs"
      className="flex h-full w-72 flex-shrink-0 flex-col border-r"
      data-testid="tariff-list"
    >
      <div className="space-y-2 border-b p-3">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search
              aria-hidden="true"
              className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              aria-label="Search tariffs"
              placeholder="Search…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              className="pl-7"
            />
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            aria-label="New tariff"
            onClick={onNew}
            disabled={!onNew}
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        {utilities.length > 0 && (
          <select
            aria-label="Filter by utility"
            value={utility}
            onChange={(e) => setUtility(e.target.value)}
            className="h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
          >
            <option value="">All utilities</option>
            {utilities.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
        )}
      </div>
      <ul className="flex-1 overflow-auto">
        {tariffs.isLoading ? (
          <li className="space-y-1 p-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-9 w-full" />
            ))}
          </li>
        ) : tariffs.isError ? (
          <li className="p-3 text-sm text-destructive">
            Failed to load tariffs.
          </li>
        ) : filtered.length === 0 ? (
          <li className="p-4 text-sm text-muted-foreground">
            No tariffs match.
          </li>
        ) : (
          filtered.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                onClick={() => onSelect(t.id)}
                aria-current={selectedId === t.id ? "true" : undefined}
                className={cn(
                  "flex w-full flex-col items-start gap-0.5 border-b px-3 py-2 text-left text-sm hover:bg-muted/40 focus-visible:bg-muted/50 focus-visible:outline-none",
                  selectedId === t.id && "bg-primary/10 text-primary",
                )}
              >
                <span className="line-clamp-1 font-medium">{t.name}</span>
                <span className="text-xs text-muted-foreground">
                  {t.utility ?? "—"}
                  {t.sector ? ` · ${t.sector}` : ""}
                </span>
              </button>
            </li>
          ))
        )}
      </ul>
    </aside>
  );
}
