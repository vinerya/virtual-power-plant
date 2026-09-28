"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Pencil, Trash2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { apiErrorMessage, deleteTariff, getTariff } from "@/lib/api/tariffs";
import type { Tariff } from "@/lib/api/tariffs";
import { NEM_LABELS, componentValue } from "./format";

const ScheduleHeatmap = dynamic(
  () => import("./schedule-heatmap").then((m) => m.ScheduleHeatmap),
  { ssr: false, loading: () => <Skeleton className="h-64 w-full" /> },
);

const BillSimulator = dynamic(
  () => import("./bill-simulator").then((m) => m.BillSimulator),
  { ssr: false, loading: () => <Skeleton className="h-72 w-full" /> },
);

export function TariffDetail({
  tariffId,
  isAdmin = false,
  onEdit,
  onDeleted,
}: {
  tariffId: string | null;
  isAdmin?: boolean;
  onEdit?: (t: Tariff) => void;
  onDeleted?: () => void;
}) {
  const qc = useQueryClient();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [dayType, setDayType] = useState<"weekday" | "weekend">("weekday");
  const q = useQuery({
    queryKey: ["tariff", tariffId],
    queryFn: () => getTariff(tariffId!),
    enabled: !!tariffId,
    staleTime: 60_000,
  });

  const del = useMutation({
    mutationFn: () => deleteTariff(tariffId!),
    onSuccess: () => {
      toast.success("Tariff deleted");
      setConfirmDelete(false);
      qc.invalidateQueries({ queryKey: ["tariffs"] });
      qc.removeQueries({ queryKey: ["tariff", tariffId] });
      onDeleted?.();
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Delete failed")),
  });

  if (!tariffId) {
    return (
      <div className="grid h-full flex-1 place-items-center p-8 text-sm text-muted-foreground">
        Select a tariff from the list to view its schedule, components, and run a simulation.
      </div>
    );
  }
  if (q.isLoading) {
    return (
      <div className="flex-1 space-y-3 p-4">
        <Skeleton className="h-7 w-1/3" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }
  if (q.isError || !q.data) {
    return (
      <div className="flex-1 p-6">
        <ErrorState title="Failed to load tariff." error={q.error} onRetry={() => q.refetch()} />
      </div>
    );
  }

  const t = q.data;
  const heatmap = dayType === "weekday" ? t.tou_heatmap : t.tou_heatmap_weekend;
  return (
    <div className="flex flex-1 flex-col overflow-auto" data-testid="tariff-detail">
      <header className="space-y-2 border-b p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h2 className="text-xl font-semibold">{t.name}</h2>
            <p className="text-xs text-muted-foreground">
              {t.utility || "—"}
              {t.sector ? ` · ${t.sector}` : ""}
              {t.source ? ` · ${t.source}` : ""}
              {t.urdb_label ? ` ${t.urdb_label}` : ""}
              {t.effective_date ? ` · effective ${t.effective_date}` : ""}
            </p>
          </div>
          {isAdmin && (
            <div className="flex shrink-0 items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => onEdit?.(t)}
                data-testid="edit-tariff"
              >
                <Pencil className="mr-1 h-3.5 w-3.5" />
                Edit
              </Button>
              {confirmDelete ? (
                <>
                  <Button
                    type="button"
                    variant="destructive"
                    size="sm"
                    onClick={() => del.mutate()}
                    disabled={del.isPending}
                    data-testid="confirm-delete-tariff"
                  >
                    {del.isPending ? "Deleting…" : "Confirm delete"}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setConfirmDelete(false)}
                  >
                    Cancel
                  </Button>
                </>
              ) : (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setConfirmDelete(true)}
                  data-testid="delete-tariff"
                >
                  <Trash2 className="mr-1 h-3.5 w-3.5" />
                  Delete
                </Button>
              )}
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">{t.is_tou ? "Time-of-use" : "Non-TOU"}</Badge>
          <Badge variant="outline" data-testid="tariff-nem">
            Exports: {NEM_LABELS[t.nem_regime] ?? t.nem_regime}
            {t.nem_source === "urdb_dgrules" ? " (URDB dgrules)" : ""}
          </Badge>
          <Badge variant="outline" className="font-mono text-[10px]">
            {t.id}
          </Badge>
        </div>
        {t.parse_error && (
          <p role="alert" className="text-sm text-destructive">
            This tariff&apos;s URDB JSON cannot be billed: {t.parse_error}
          </p>
        )}
      </header>
      <Tabs defaultValue="schedule" className="flex-1 p-4">
        <TabsList>
          <TabsTrigger value="schedule">Schedule</TabsTrigger>
          <TabsTrigger value="components">Components</TabsTrigger>
          <TabsTrigger value="simulate">Simulate</TabsTrigger>
          <TabsTrigger value="json">URDB JSON</TabsTrigger>
        </TabsList>
        <TabsContent value="schedule">
          {t.tou_heatmap && t.tou_heatmap.length === 12 ? (
            <div className="space-y-3">
              <div
                className="inline-flex rounded-md border p-0.5 text-xs"
                role="group"
                aria-label="Day type"
              >
                {(["weekday", "weekend"] as const).map((d) => (
                  <button
                    key={d}
                    type="button"
                    aria-pressed={dayType === d}
                    onClick={() => setDayType(d)}
                    className={
                      "rounded px-2 py-1 " +
                      (dayType === d ? "bg-primary text-primary-foreground" : "")
                    }
                  >
                    {d === "weekday" ? "Weekdays" : "Weekends & holidays"}
                  </button>
                ))}
              </div>
              {heatmap && heatmap.length === 12 ? (
                <ScheduleHeatmap heatmap={heatmap} />
              ) : (
                <p className="text-sm text-muted-foreground">No schedule for this day type.</p>
              )}
              <p className="text-xs text-muted-foreground">
                Import energy rate by month and local hour (tier 1 where tiered). Demand,
                fixed and other charges are listed under Components.
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground" data-testid="no-schedule">
              This tariff has no energy schedule.
            </p>
          )}
        </TabsContent>
        <TabsContent value="components">
          {t.components.length === 0 ? (
            <p className="text-sm text-muted-foreground">No billable components.</p>
          ) : (
            <ul className="divide-y rounded-md border" data-testid="components-list">
              {t.components.map((c, i) => (
                <li key={i} className="flex items-start justify-between gap-4 p-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium">{c.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {c.kind}
                      {c.detail ? ` · ${c.detail}` : ""}
                      {c.sell_rate != null ? ` · export ${c.sell_rate.toFixed(4)} $/kWh` : ""}
                    </p>
                    {c.schedule && c.schedule.length > 0 && (
                      <ul className="mt-1 text-xs text-muted-foreground">
                        {c.schedule.map((line) => (
                          <li key={line}>{line}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                  <span className="shrink-0 text-right font-mono text-sm tabular-nums">
                    {componentValue(c)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </TabsContent>
        <TabsContent value="simulate">
          <BillSimulator tariff={t} />
        </TabsContent>
        <TabsContent value="json">
          {t.description && (
            <p className="mb-2 text-xs text-muted-foreground">{t.description}</p>
          )}
          <pre className="max-h-[60vh] overflow-auto rounded-md border bg-muted/40 p-3 text-xs">
            {JSON.stringify(t.urdb_json, null, 2)}
          </pre>
        </TabsContent>
      </Tabs>
    </div>
  );
}
