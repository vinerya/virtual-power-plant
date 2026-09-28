"use client";

import { useEffect, useMemo, useState } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { toast } from "sonner";
import { ShieldCheck, ShieldOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  ackAlert,
  bulkAckAlerts,
  listAlerts,
  snoozeAlert,
} from "@/lib/api/alerts";
import { getWsClient } from "@/lib/ws/client";
import type { Alert, AlertSeverity } from "@/lib/api/types";
import { cn } from "@/lib/utils";
import { SeveritySparkline } from "./severity-sparkline";
import { AlertRow } from "./alert-row";
import { ErrorState } from "@/components/ui/error-state";

const FILTERS: { key: AlertSeverity | "all"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "critical", label: "Critical" },
  { key: "warning", label: "Warning" },
  { key: "info", label: "Info" },
];

const PULSE_MS = 2500;

export function AlertsList() {
  const qc = useQueryClient();
  const [filter, setFilter] = useState<AlertSeverity | "all">("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [pulses, setPulses] = useState<Record<string, number>>({});

  const since = useMemo(() => {
    const d = new Date(Date.now() - 24 * 60 * 60 * 1000);
    return d.toISOString();
  }, []);

  const q = useQuery({
    queryKey: ["alerts", since],
    queryFn: () => listAlerts({ since }),
    refetchInterval: 15_000,
  });

  // ---------- WebSocket live tail ----------
  useEffect(() => {
    const client = getWsClient();
    const off = client.on((msg) => {
      if (msg.channel !== "alerts") return;
      const data = msg.data as Partial<Alert> | null;
      if (!data || !data.id) {
        // No id — refetch the list rather than guess.
        qc.invalidateQueries({ queryKey: ["alerts"] });
        return;
      }
      qc.setQueryData<Alert[] | undefined>(["alerts", since], (prev) => {
        const arr = prev ?? [];
        const without = arr.filter((a) => a.id !== data.id);
        const merged = { ...(arr.find((a) => a.id === data.id) ?? {}), ...data } as Alert;
        return [merged, ...without];
      });
      setPulses((p) => ({ ...p, [data.id as string]: Date.now() + PULSE_MS }));
    });
    return off;
  }, [qc, since]);

  // ---------- Filtering & visibility ----------
  const visible = useMemo(() => {
    const list = q.data ?? [];
    return list
      .filter((a) => filter === "all" || a.severity === filter)
      .sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
      );
  }, [q.data, filter]);

  const counts = useMemo(() => {
    const list = q.data ?? [];
    const c = { critical: 0, warning: 0, info: 0, active: 0 };
    for (const a of list) {
      c[a.severity] += 1;
      if (a.status === "active") c.active += 1;
    }
    return c;
  }, [q.data]);

  // ---------- Mutations ----------
  const ackM = useMutation({
    mutationFn: (id: string) => ackAlert(id),
    onMutate: (id) => optimisticUpdate(qc, since, id, { status: "acknowledged" }),
    onError: () => toast.error("Failed to acknowledge"),
    onSuccess: () => {
      toast.success("Alert acknowledged");
      qc.invalidateQueries({ queryKey: ["alerts"] });
    },
  });

  const snoozeM = useMutation({
    mutationFn: ({ id, ms }: { id: string; ms: number }) => snoozeAlert(id, ms),
    onMutate: ({ id, ms }) =>
      optimisticUpdate(qc, since, id, {
        status: "snoozed",
        snoozed_until: new Date(Date.now() + ms).toISOString(),
      }),
    onError: () => toast.error("Failed to snooze"),
    onSuccess: () => {
      toast.success("Alert snoozed");
      qc.invalidateQueries({ queryKey: ["alerts"] });
    },
  });

  const bulkAckM = useMutation({
    mutationFn: (ids: string[]) => bulkAckAlerts(ids),
    onSuccess: (_d, ids) => {
      toast.success(`Acknowledged ${ids.length} alert${ids.length === 1 ? "" : "s"}`);
      setSelected(new Set());
      qc.invalidateQueries({ queryKey: ["alerts"] });
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Bulk ack failed");
      qc.invalidateQueries({ queryKey: ["alerts"] });
    },
  });

  // ---------- Selection helpers ----------
  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const selectAllVisible = () => {
    setSelected(new Set(visible.filter((a) => a.status === "active").map((a) => a.id)));
  };

  return (
    <div className="space-y-4" data-testid="alerts-list">
      <Card>
        <CardHeader className="flex flex-row items-baseline justify-between space-y-0">
          <div>
            <CardTitle>Volume — last 24 hours</CardTitle>
            <p className="text-xs text-muted-foreground">
              {counts.active} active · {counts.critical} critical ·{" "}
              {counts.warning} warning · {counts.info} info
            </p>
          </div>
          {q.isFetching && (
            <span className="text-xs text-muted-foreground">refreshing…</span>
          )}
        </CardHeader>
        <CardContent>
          <SeveritySparkline alerts={q.data ?? []} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 space-y-0">
          <div className="flex flex-wrap items-center gap-2">
            {FILTERS.map((f) => {
              const active = filter === f.key;
              const n =
                f.key === "all" ? (q.data?.length ?? 0) : counts[f.key];
              return (
                <button
                  key={f.key}
                  type="button"
                  onClick={() => setFilter(f.key)}
                  data-testid={`filter-${f.key}`}
                  aria-pressed={active}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                    active
                      ? "border-primary bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted",
                  )}
                >
                  {f.label}
                  <Badge variant="outline" className="text-[10px]">
                    {n}
                  </Badge>
                </button>
              );
            })}
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={selectAllVisible}
              disabled={visible.length === 0}
            >
              Select active
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={() => bulkAckM.mutate(Array.from(selected))}
              disabled={selected.size === 0 || bulkAckM.isPending}
              data-testid="bulk-ack"
            >
              <ShieldCheck className="mr-1 h-3.5 w-3.5" /> Ack {selected.size}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="px-0">
          {q.isLoading ? (
            <div className="space-y-2 px-3 py-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-14 w-full" />
              ))}
            </div>
          ) : q.isError ? (
            <ErrorState
              className="mx-3 my-3"
              title="Failed to load alerts."
              error={q.error}
              onRetry={() => q.refetch()}
            />
          ) : visible.length === 0 ? (
            <EmptyState />
          ) : (
            <ul role="rowgroup" data-testid="alert-rows">
              {visible.map((a) => (
                <AlertRow
                  key={a.id}
                  alert={a}
                  selected={selected.has(a.id)}
                  onToggleSelect={() => toggle(a.id)}
                  onAck={() => ackM.mutate(a.id)}
                  onSnooze={(ms) => snoozeM.mutate({ id: a.id, ms })}
                  pulseUntil={pulses[a.id]}
                />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function optimisticUpdate(
  qc: ReturnType<typeof useQueryClient>,
  since: string,
  id: string,
  patch: Partial<Alert>,
) {
  qc.setQueryData<Alert[] | undefined>(["alerts", since], (prev) =>
    prev?.map((a) => (a.id === id ? { ...a, ...patch } : a)),
  );
}

function EmptyState() {
  return (
    <div
      className="flex flex-col items-center justify-center gap-2 p-10 text-center"
      data-testid="alerts-empty"
    >
      <ShieldOff className="h-10 w-10 text-emerald-500" aria-hidden="true" />
      <p className="text-base font-medium">All clear</p>
      <p className="text-xs text-muted-foreground">
        No alerts match the current filter in the last 24 hours.
      </p>
    </div>
  );
}
