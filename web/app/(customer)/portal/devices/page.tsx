"use client";

import { useQuery } from "@tanstack/react-query";
import { Battery, Car, Thermometer } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getMyDevices } from "@/lib/api/customer";
import type { CustomerDevice } from "@/lib/api/types";
import { formatPercent, formatPower } from "@/lib/utils";

export default function CustomerDevicesPage() {
  const q = useQuery({
    queryKey: ["customer", "devices"],
    queryFn: getMyDevices,
    refetchInterval: 15_000,
  });

  return (
    <div className="space-y-6" data-testid="devices-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Your devices</h2>
        <p className="text-sm text-muted-foreground">
          Read-only state of devices enrolled in your account.
        </p>
      </div>

      {q.isLoading ? (
        <div className="grid gap-3 md:grid-cols-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
      ) : q.isError ? (
        <p className="text-sm text-destructive">Failed to load devices.</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {(q.data ?? []).map((d) => (
            <DeviceCard key={d.id} d={d} />
          ))}
        </div>
      )}
    </div>
  );
}

function DeviceCard({ d }: { d: CustomerDevice }) {
  const Icon =
    d.kind === "battery" ? Battery : d.kind === "ev" ? Car : Thermometer;
  return (
    <Card data-testid="device-card" data-device-kind={d.kind}>
      <CardHeader className="flex flex-row items-center gap-2 space-y-0 pb-2">
        <Icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <CardTitle>{d.name}</CardTitle>
        <Badge variant="outline" className="ml-auto capitalize">
          {d.state}
        </Badge>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          {typeof d.current_power === "number" && (
            <KV label="Power" value={formatPower(d.current_power)} />
          )}
          {typeof d.state_of_charge === "number" && (
            <KV label="SOC" value={formatPercent(d.state_of_charge)} />
          )}
          {typeof d.setpoint_c === "number" && (
            <KV label="Setpoint" value={`${d.setpoint_c.toFixed(0)}°C`} />
          )}
          <KV label="Type" value={d.kind} />
        </dl>
      </CardContent>
    </Card>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}
