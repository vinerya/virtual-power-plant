"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { listResources } from "@/lib/api/resources";
import type { ResourceResponse } from "@/lib/api/types";
import { cn, formatPower, formatRelativeTime } from "@/lib/utils";

type SortDir = "asc" | "desc";

export function FleetOverview() {
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const query = useQuery({
    queryKey: ["resources"],
    queryFn: listResources,
    refetchInterval: 5_000,
  });

  const resources = query.data ?? [];

  const stats = useMemo(() => {
    const total = resources.length;
    const online = resources.filter((r) => r.online).length;
    const rated = resources.reduce((s, r) => s + r.rated_power, 0);
    const current = resources.reduce((s, r) => s + r.current_power, 0);
    return { total, online, rated, current };
  }, [resources]);

  const sorted = useMemo(() => {
    const list = [...resources];
    list.sort((a, b) =>
      sortDir === "asc"
        ? a.current_power - b.current_power
        : b.current_power - a.current_power,
    );
    return list;
  }, [resources, sortDir]);

  const lastUpdated = query.dataUpdatedAt
    ? new Date(query.dataUpdatedAt)
    : null;

  return (
    <div className="space-y-6">
      <section
        aria-label="Fleet stats"
        className="grid grid-cols-2 gap-4 md:grid-cols-4"
      >
        <StatCard
          label="Resources"
          value={query.isLoading ? null : `${stats.total}`}
          testid="stat-total"
        />
        <StatCard
          label="Online"
          value={
            query.isLoading
              ? null
              : `${stats.online} / ${stats.total || 0}`
          }
          testid="stat-online"
        />
        <StatCard
          label="Rated power"
          value={query.isLoading ? null : formatPower(stats.rated)}
          testid="stat-rated"
        />
        <StatCard
          label="Current power"
          value={query.isLoading ? null : formatPower(stats.current)}
          testid="stat-current"
        />
      </section>

      <section aria-label="Fleet resources">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <div>
              <h2 className="text-base font-semibold">Resources</h2>
              <p className="text-sm text-muted-foreground">
                {lastUpdated
                  ? `Last refresh ${formatRelativeTime(lastUpdated)}`
                  : "Loading…"}
              </p>
            </div>
            {query.isFetching && (
              <span className="text-xs text-muted-foreground">refreshing…</span>
            )}
          </CardHeader>
          <CardContent className="px-0">
            {query.isLoading ? (
              <TableSkeleton />
            ) : query.isError ? (
              <p className="px-6 py-8 text-sm text-destructive">
                Could not load resources. Check that the FastAPI backend is
                reachable.
              </p>
            ) : sorted.length === 0 ? (
              <p
                className="px-6 py-8 text-sm text-muted-foreground"
                data-testid="empty-state"
              >
                No resources registered yet. Use the API or CLI to add a
                battery, solar array, or wind turbine.
              </p>
            ) : (
              <ResourceTable
                rows={sorted}
                sortDir={sortDir}
                onToggleSort={() =>
                  setSortDir((d) => (d === "asc" ? "desc" : "asc"))
                }
              />
            )}
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  testid,
}: {
  label: string;
  value: string | null;
  testid?: string;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>{label}</CardTitle>
      </CardHeader>
      <CardContent>
        {value === null ? (
          <Skeleton className="h-7 w-24" />
        ) : (
          <p className="text-2xl font-semibold tracking-tight" data-testid={testid}>
            {value}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function ResourceTable({
  rows,
  sortDir,
  onToggleSort,
}: {
  rows: ResourceResponse[];
  sortDir: SortDir;
  onToggleSort: () => void;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Name
            </th>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Type
            </th>
            <th scope="col" className="px-6 py-2 text-right font-medium">
              Rated
            </th>
            <th scope="col" className="px-6 py-2 text-right font-medium">
              <button
                type="button"
                onClick={onToggleSort}
                className="inline-flex items-center gap-1 hover:text-foreground"
                aria-label={`Sort by current power ${sortDir === "asc" ? "descending" : "ascending"}`}
              >
                Current
                {sortDir === "asc" ? (
                  <ArrowUp className="h-3 w-3" />
                ) : (
                  <ArrowDown className="h-3 w-3" />
                )}
              </button>
            </th>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Status
            </th>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Updated
            </th>
          </tr>
        </thead>
        <tbody data-testid="resource-rows">
          {rows.map((r) => (
            <tr
              key={r.id}
              className="border-b last:border-b-0 hover:bg-muted/30"
            >
              <td className="px-6 py-3 font-medium">{r.name}</td>
              <td className="px-6 py-3 text-muted-foreground">
                {r.resource_type.replace("_", " ")}
              </td>
              <td className="px-6 py-3 text-right tabular-nums">
                {formatPower(r.rated_power)}
              </td>
              <td
                className={cn(
                  "px-6 py-3 text-right tabular-nums",
                  r.current_power > 0 && "text-success",
                  r.current_power < 0 && "text-destructive",
                )}
              >
                {formatPower(r.current_power)}
              </td>
              <td className="px-6 py-3">
                {r.online ? (
                  <Badge variant="success">online</Badge>
                ) : (
                  <Badge variant="secondary">offline</Badge>
                )}
              </td>
              <td className="px-6 py-3 text-muted-foreground">
                {formatRelativeTime(r.updated_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TableSkeleton() {
  return (
    <div className="space-y-2 px-6 py-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  );
}
