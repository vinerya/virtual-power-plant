"use client";

import dynamic from "next/dynamic";
import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { getTariff } from "@/lib/api/tariffs";

const ScheduleHeatmap = dynamic(
  () =>
    import("./schedule-heatmap").then((m) => m.ScheduleHeatmap),
  { ssr: false, loading: () => <Skeleton className="h-64 w-full" /> },
);

const BillSimulator = dynamic(
  () => import("./bill-simulator").then((m) => m.BillSimulator),
  { ssr: false, loading: () => <Skeleton className="h-72 w-full" /> },
);

export function TariffDetail({ tariffId }: { tariffId: string | null }) {
  const q = useQuery({
    queryKey: ["tariff", tariffId],
    queryFn: () => getTariff(tariffId!),
    enabled: !!tariffId,
    staleTime: 60_000,
  });

  if (!tariffId) {
    return (
      <div className="grid h-full place-items-center p-8 text-sm text-muted-foreground">
        Select a tariff from the list to view its schedule, components, and run a simulation.
      </div>
    );
  }
  if (q.isLoading) {
    return (
      <div className="space-y-3 p-4">
        <Skeleton className="h-7 w-1/3" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }
  if (q.isError || !q.data) {
    return (
      <div className="p-6 text-sm text-destructive">
        Failed to load tariff.
      </div>
    );
  }

  const t = q.data;
  return (
    <div className="flex flex-1 flex-col" data-testid="tariff-detail">
      <header className="border-b p-4">
        <div className="flex items-baseline justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold">{t.name}</h2>
            <p className="text-xs text-muted-foreground">
              {t.utility ?? "—"} {t.sector ? ` · ${t.sector}` : ""}
              {t.source ? ` · ${t.source}` : ""}
            </p>
          </div>
          <Badge variant="outline" className="font-mono text-[10px]">
            {t.id}
          </Badge>
        </div>
      </header>
      <Tabs defaultValue="schedule" className="flex-1 p-4">
        <TabsList>
          <TabsTrigger value="schedule">Schedule</TabsTrigger>
          <TabsTrigger value="components">Components</TabsTrigger>
          <TabsTrigger value="simulate">Simulate</TabsTrigger>
        </TabsList>
        <TabsContent value="schedule">
          {t.tou_heatmap && t.tou_heatmap.length === 12 ? (
            <ScheduleHeatmap heatmap={t.tou_heatmap} />
          ) : (
            <p className="text-sm text-muted-foreground">
              No TOU schedule available for this tariff.
            </p>
          )}
        </TabsContent>
        <TabsContent value="components">
          <ul className="divide-y rounded-md border" data-testid="components-list">
            {t.components.map((c, i) => (
              <li key={i} className="flex items-baseline justify-between p-3">
                <div>
                  <p className="text-sm font-medium">{c.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {c.kind}
                    {c.unit ? ` · ${c.unit}` : ""}
                  </p>
                </div>
                <span className="font-mono text-sm tabular-nums">
                  {c.rate != null
                    ? c.rate.toFixed(3)
                    : c.rates
                      ? `[${c.rates.map((r) => r.toFixed(3)).join(", ")}]`
                      : "—"}
                </span>
              </li>
            ))}
          </ul>
        </TabsContent>
        <TabsContent value="simulate">
          <BillSimulator tariffId={t.id} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
