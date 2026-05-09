"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { SiteList } from "@/components/sites/site-list";
import { listSites } from "@/lib/api/sites";
import { formatPower } from "@/lib/utils";

// MapLibre is heavy (~500KB). Lazy-load with SSR off.
const SiteMap = dynamic(
  () => import("@/components/sites/site-map").then((m) => m.SiteMap),
  {
    ssr: false,
    loading: () => <Skeleton className="h-full w-full" />,
  },
);

export default function SitesPage() {
  const q = useQuery({
    queryKey: ["sites"],
    queryFn: listSites,
    refetchInterval: 30_000,
  });

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [flyToId, setFlyToId] = useState<string | null>(null);

  const [layers, setLayers] = useState({
    heatmap: true,
    capacity: true,
    region: false,
  });

  const sites = q.data ?? [];

  const kpi = useMemo(() => {
    const totalKw = sites.reduce((s, x) => s + x.current_power, 0);
    const totalRes = sites.reduce((s, x) => s + x.total_resources, 0);
    const totalAlerts = sites.reduce((s, x) => s + x.active_alerts, 0);
    return {
      sites: sites.length,
      resources: totalRes,
      power: totalKw,
      alerts: totalAlerts,
    };
  }, [sites]);

  const handleSelect = (id: string) => {
    setSelectedId(id);
    setFlyToId(id);
  };

  return (
    <div className="flex h-[calc(100vh-7rem)] flex-col gap-4" data-testid="sites-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Sites</h2>
        <p className="text-sm text-muted-foreground">
          Geographic view of every operating site.
        </p>
      </div>

      <section
        aria-label="Sites KPIs"
        className="grid grid-cols-2 gap-3 md:grid-cols-4"
      >
        <Kpi label="Total sites" value={`${kpi.sites}`} />
        <Kpi label="Total resources" value={`${kpi.resources}`} />
        <Kpi label="Aggregate power" value={formatPower(kpi.power)} />
        <Kpi label="Active alerts" value={`${kpi.alerts}`} />
      </section>

      <div className="flex flex-1 overflow-hidden rounded-md border bg-card">
        {q.isLoading ? (
          <div className="flex flex-1 items-center justify-center p-6">
            <Skeleton className="h-full w-full" />
          </div>
        ) : q.isError ? (
          <p className="p-6 text-sm text-destructive">Failed to load sites.</p>
        ) : (
          <>
            <SiteList
              sites={sites}
              selectedId={selectedId}
              onSelect={handleSelect}
            />
            <div className="relative flex flex-1 flex-col">
              <div className="absolute left-3 top-3 z-10 flex flex-wrap gap-2 rounded-md border bg-background/90 px-2 py-1.5 text-xs shadow backdrop-blur">
                <ToggleChip
                  label="Heatmap"
                  on={layers.heatmap}
                  onChange={(v) => setLayers((l) => ({ ...l, heatmap: v }))}
                  testid="layer-heatmap"
                />
                <ToggleChip
                  label="Capacity bubbles"
                  on={layers.capacity}
                  onChange={(v) => setLayers((l) => ({ ...l, capacity: v }))}
                  testid="layer-capacity"
                />
                <ToggleChip
                  label="Regions"
                  on={layers.region}
                  onChange={(v) => setLayers((l) => ({ ...l, region: v }))}
                  testid="layer-region"
                />
              </div>
              <div className="flex-1">
                <SiteMap
                  sites={sites}
                  selectedId={selectedId}
                  onSelect={handleSelect}
                  flyToId={flyToId}
                  layers={layers}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-semibold tracking-tight tabular-nums">
          {value}
        </p>
      </CardContent>
    </Card>
  );
}

function ToggleChip({
  label,
  on,
  onChange,
  testid,
}: {
  label: string;
  on: boolean;
  onChange: (v: boolean) => void;
  testid?: string;
}) {
  return (
    <label
      className="flex cursor-pointer items-center gap-1.5"
      data-testid={testid}
    >
      <input
        type="checkbox"
        checked={on}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3.5 w-3.5"
      />
      <span>{label}</span>
    </label>
  );
}
