"use client";

import { useEffect, useMemo, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { LineSeriesChart } from "@/components/charts/lazy-line-chart";
import { getResource } from "@/lib/api/resources";
import { getResourceMetrics } from "@/lib/api/metrics";
import type {
  ResourceMetricsPoint,
  ResourceResponse,
} from "@/lib/api/types";
import {
  formatDateTime,
  formatPercent,
  formatPower,
  formatRelativeTime,
} from "@/lib/utils";
import { SubtypePanel } from "./subtype-panel";

const POLL_MS = 10_000;
const BUFFER_LIMIT = 24 * 60; // 24h @ 1min sampling cap.

type Buffer = ResourceMetricsPoint[];

export function AssetDetail({ id }: { id: string }) {
  const resourceQuery = useQuery({
    queryKey: ["resource", id],
    queryFn: () => getResource(id),
    refetchInterval: POLL_MS,
  });

  const metricsQuery = useQuery({
    queryKey: ["resource-metrics", id, "24h"],
    queryFn: () => getResourceMetrics(id, "24h"),
    staleTime: 30_000,
  });

  // ----------------------------------------------------------------------
  // Fallback: when the backend has no metrics endpoint (returns null/404),
  // build a rolling 24h buffer client-side from the polled snapshot. Each
  // resource refresh appends one point; we keep the most recent BUFFER_LIMIT
  // entries. This is intentionally documented (see comment above) and will
  // be replaced when the backend exposes `/resources/{id}/metrics`.
  // ----------------------------------------------------------------------
  const bufferRef = useRef<Buffer>([]);
  useEffect(() => {
    if (metricsQuery.data) return; // Server data wins.
    const snap = resourceQuery.data;
    if (!snap) return;
    const last = bufferRef.current[bufferRef.current.length - 1];
    if (last && last.timestamp === snap.updated_at) return;
    bufferRef.current = [
      ...bufferRef.current,
      {
        timestamp: snap.updated_at,
        power: snap.current_power,
        state_of_charge:
          typeof snap.state_of_charge === "number"
            ? snap.state_of_charge
            : undefined,
        efficiency: snap.efficiency ?? undefined,
      },
    ].slice(-BUFFER_LIMIT);
  }, [metricsQuery.data, resourceQuery.data]);

  const series = useMemo(() => {
    const points: ResourceMetricsPoint[] =
      metricsQuery.data?.points ?? bufferRef.current;
    const isBattery = resourceQuery.data?.resource_type === "battery";
    return points.map((p) => ({
      timestamp: p.timestamp,
      value: isBattery
        ? typeof p.state_of_charge === "number"
          ? p.state_of_charge * 100
          : p.power
        : p.power,
    }));
  }, [metricsQuery.data, resourceQuery.data, resourceQuery.dataUpdatedAt]);

  if (resourceQuery.isLoading) return <DetailSkeleton />;
  if (resourceQuery.isError || !resourceQuery.data) {
    return (
      <Card>
        <CardContent className="p-6 text-sm text-destructive">
          Could not load resource. It may have been removed or the backend is
          unreachable.
        </CardContent>
      </Card>
    );
  }

  const r = resourceQuery.data;
  const isBattery = r.resource_type === "battery";
  const chartUnit = isBattery ? "%" : "kW";

  return (
    <div className="space-y-6" data-testid="asset-detail">
      <Header r={r} />
      <Stats r={r} />
      <Card>
        <CardHeader>
          <CardTitle>
            {isBattery ? "State of charge — last 24h" : "Power output — last 24h"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {series.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No samples yet — the live buffer fills as snapshots arrive (one
              per {POLL_MS / 1000}s).
            </p>
          ) : (
            <LineSeriesChart data={series} unit={chartUnit} />
          )}
          {!metricsQuery.data && (
            <p className="mt-2 text-xs text-muted-foreground">
              Live client-side buffer (backend metrics endpoint not available).
            </p>
          )}
        </CardContent>
      </Card>
      <SubtypePanel r={r} />
    </div>
  );
}

function Header({ r }: { r: ResourceResponse }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <h2 className="text-2xl font-semibold tracking-tight" data-testid="asset-name">
        {r.name}
      </h2>
      <Badge variant="outline" className="capitalize">
        {String(r.resource_type).replace("_", " ")}
      </Badge>
      {r.online ? (
        <Badge variant="success">online</Badge>
      ) : (
        <Badge variant="secondary">offline</Badge>
      )}
    </div>
  );
}

function Stats({ r }: { r: ResourceResponse }) {
  return (
    <section
      aria-label="Asset stats"
      className="grid grid-cols-2 gap-4 md:grid-cols-4"
    >
      <Stat label="Rated power" value={formatPower(r.rated_power)} />
      <Stat label="Current power" value={formatPower(r.current_power)} />
      <Stat
        label="Efficiency"
        value={r.efficiency != null ? formatPercent(r.efficiency) : "—"}
      />
      <Stat
        label="Last update"
        value={formatRelativeTime(r.updated_at)}
        title={formatDateTime(r.updated_at)}
      />
    </section>
  );
}

function Stat({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <p
          className="text-2xl font-semibold tracking-tight"
          title={title}
        >
          {value}
        </p>
      </CardContent>
    </Card>
  );
}

function DetailSkeleton() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-8 w-64" />
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
      <Skeleton className="h-72 w-full" />
      <Skeleton className="h-48 w-full" />
    </div>
  );
}
