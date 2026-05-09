"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { listDispatches } from "@/lib/api/dispatch";
import { listResources } from "@/lib/api/resources";
import type { DispatchRun } from "@/lib/api/types";
import { formatDateTime } from "@/lib/utils";
import { DispatchSheet } from "./dispatch-sheet";

type Window = "24h" | "7d" | "30d" | "custom";

const PAGE_SIZE = 25;

export function DispatchesView() {
  const [windowSel, setWindowSel] = useState<Window>("24h");
  const [customStart, setCustomStart] = useState<string>("");
  const [customEnd, setCustomEnd] = useState<string>("");
  const [resourceFilter, setResourceFilter] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<DispatchRun | null>(null);

  const range = useMemo(() => computeRange(windowSel, customStart, customEnd), [
    windowSel,
    customStart,
    customEnd,
  ]);

  const resources = useQuery({
    queryKey: ["resources"],
    queryFn: listResources,
    staleTime: 60_000,
  });

  const dispatches = useQuery({
    queryKey: ["dispatches", { range, resourceFilter }],
    queryFn: () =>
      listDispatches({
        start: range.start,
        end: range.end,
        resource_ids: resourceFilter.length ? resourceFilter : undefined,
        limit: 200,
      }),
  });

  const rows = dispatches.data ?? [];
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="space-y-6" data-testid="dispatches-view">
      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-[1fr_1fr_2fr]">
            <div className="space-y-1.5">
              <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Window
              </label>
              <Select
                aria-label="Time window"
                value={windowSel}
                onChange={(e) => {
                  setWindowSel(e.target.value as Window);
                  setPage(0);
                }}
                data-testid="window-select"
              >
                <option value="24h">Last 24 hours</option>
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="custom">Custom range</option>
              </Select>
            </div>
            {windowSel === "custom" ? (
              <div className="md:col-span-1 grid grid-cols-2 gap-2">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    From
                  </label>
                  <input
                    type="datetime-local"
                    value={customStart}
                    onChange={(e) => setCustomStart(e.target.value)}
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    To
                  </label>
                  <input
                    type="datetime-local"
                    value={customEnd}
                    onChange={(e) => setCustomEnd(e.target.value)}
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  />
                </div>
              </div>
            ) : (
              <div />
            )}
            <div className="space-y-1.5">
              <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Resources
              </label>
              <select
                multiple
                aria-label="Resources"
                className="h-24 w-full rounded-md border border-input bg-background px-2 py-1 text-sm"
                value={resourceFilter}
                onChange={(e) => {
                  const opts = Array.from(e.target.selectedOptions).map(
                    (o) => o.value,
                  );
                  setResourceFilter(opts);
                  setPage(0);
                }}
                data-testid="resource-multiselect"
              >
                {(resources.data ?? []).map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">
                Hold ⌘/Ctrl to select multiple. Empty = all.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle>Dispatch history</CardTitle>
            <p className="text-sm text-muted-foreground">
              {dispatches.isLoading
                ? "Loading…"
                : `${rows.length} run${rows.length === 1 ? "" : "s"} in window`}
            </p>
          </div>
        </CardHeader>
        <CardContent className="px-0">
          {dispatches.isLoading ? (
            <div className="space-y-2 px-6 py-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : dispatches.isError ? (
            <p className="px-6 py-8 text-sm text-destructive">
              Failed to load dispatch history.
            </p>
          ) : rows.length === 0 ? (
            <p className="px-6 py-8 text-sm text-muted-foreground">
              No dispatch runs in this window.
            </p>
          ) : (
            <DispatchTable rows={pageRows} onSelect={setSelected} />
          )}
        </CardContent>
        {pages > 1 && (
          <div className="flex items-center justify-between border-t px-6 py-3 text-sm">
            <span className="text-muted-foreground">
              Page {page + 1} / {pages}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
                disabled={page >= pages - 1}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </Card>

      <DispatchSheet
        run={selected}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}

function DispatchTable({
  rows,
  onSelect,
}: {
  rows: DispatchRun[];
  onSelect: (r: DispatchRun) => void;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Timestamp
            </th>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Type
            </th>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Status
            </th>
            <th scope="col" className="px-6 py-2 text-right font-medium">
              Objective
            </th>
            <th scope="col" className="px-6 py-2 text-right font-medium">
              Solve (ms)
            </th>
            <th scope="col" className="px-6 py-2 text-left font-medium">
              Fallback
            </th>
          </tr>
        </thead>
        <tbody data-testid="dispatch-rows">
          {rows.map((r) => (
            <tr
              key={r.id}
              tabIndex={0}
              role="button"
              aria-label={`Open dispatch ${r.id}`}
              onClick={() => onSelect(r)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(r);
                }
              }}
              className="cursor-pointer border-b last:border-b-0 hover:bg-muted/30 focus-visible:bg-muted/40 focus-visible:outline-none"
            >
              <td className="px-6 py-3 tabular-nums">
                {formatDateTime(r.created_at)}
              </td>
              <td className="px-6 py-3 text-muted-foreground">
                {r.problem_type}
              </td>
              <td className="px-6 py-3">
                <StatusBadge status={r.status} />
              </td>
              <td className="px-6 py-3 text-right tabular-nums">
                {r.objective_value != null ? r.objective_value.toFixed(2) : "—"}
              </td>
              <td className="px-6 py-3 text-right tabular-nums">
                {r.solve_time_ms != null ? r.solve_time_ms.toFixed(1) : "—"}
              </td>
              <td className="px-6 py-3">
                {r.fallback_used ? (
                  <Badge variant="secondary">fallback</Badge>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase();
  if (s === "optimal" || s === "success" || s === "ok")
    return <Badge variant="success">{status}</Badge>;
  if (s === "infeasible" || s === "error" || s === "failed")
    return <Badge variant="destructive">{status}</Badge>;
  return <Badge variant="outline">{status}</Badge>;
}

function computeRange(
  win: Window,
  customStart: string,
  customEnd: string,
): { start: string; end: string } {
  const now = Date.now();
  if (win === "custom") {
    return {
      start: customStart ? new Date(customStart).toISOString() : new Date(0).toISOString(),
      end: customEnd ? new Date(customEnd).toISOString() : new Date().toISOString(),
    };
  }
  const days = win === "24h" ? 1 : win === "7d" ? 7 : 30;
  return {
    start: new Date(now - days * 24 * 3600 * 1000).toISOString(),
    end: new Date(now).toISOString(),
  };
}
