"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { FlaskConical, Plug, PlugZap, Radio } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import {
  OPERATIONAL_STATUSES,
  connectProtocol,
  disconnectProtocol,
  listProtocols,
  type ProtocolInfo,
} from "@/lib/api/protocols";
import { apiErrorMessage } from "@/lib/api/errors";
import { useRole } from "@/lib/api/session";
import { cn } from "@/lib/utils";

const KEY = ["protocols"] as const;

function formatUptime(s: number): string {
  if (!Number.isFinite(s) || s <= 0) return "—";
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  return `${m}m ${Math.floor(s % 60)}s`;
}

export function ProtocolsView() {
  const { canOperate } = useRole();
  const q = useQuery({ queryKey: KEY, queryFn: listProtocols, refetchInterval: 10_000 });

  if (q.isLoading) return <Skeleton className="h-48 w-full" />;
  if (q.error || !q.data)
    return (
      <ErrorState
        title="Failed to load protocol adapters."
        error={q.error}
        onRetry={() => void q.refetch()}
      />
    );

  if (q.data.length === 0) {
    return (
      <Card data-testid="protocols-empty">
        <CardContent className="space-y-2 p-6 text-sm">
          <p className="font-medium">No protocol adapters are running.</p>
          <p className="text-muted-foreground">
            Adapters are opt-in because they dial out to (or accept connections from) real
            equipment. Enable them in the API environment, for example{" "}
            <code>VPP_MQTT_INGESTION_ENABLED</code>, <code>VPP_MODBUS_INGESTION_ENABLED</code>,{" "}
            <code>VPP_OCPP_ENABLED</code>, <code>VPP_OPENADR_ENABLED</code> or{" "}
            <code>VPP_IEEE2030_5_ENABLED</code>, then restart the API.
          </p>
        </CardContent>
      </Card>
    );
  }

  const live = q.data.filter((p) => p.mode === "live").length;
  const simulated = q.data.length - live;

  return (
    <div className="space-y-4" data-testid="protocols-view">
      <p className="text-sm text-muted-foreground" aria-live="polite">
        {q.data.length} adapter{q.data.length === 1 ? "" : "s"}: {live} live, {simulated}{" "}
        simulated. Simulated adapters run in memory only and send nothing to real devices.
      </p>
      <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3" aria-label="Protocol adapters">
        {q.data.map((p) => (
          <li key={p.name}>
            <ProtocolCard p={p} canOperate={canOperate} />
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ModeBadge({ mode }: { mode: string }) {
  const simulated = mode === "simulated";
  return (
    <span
      data-testid="mode-badge"
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
        simulated
          ? "border-amber-400 bg-amber-50 text-amber-900 dark:bg-amber-950 dark:text-amber-200"
          : "border-emerald-500 bg-emerald-50 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
      )}
    >
      {simulated ? (
        <FlaskConical className="h-3 w-3" aria-hidden="true" />
      ) : (
        <Radio className="h-3 w-3" aria-hidden="true" />
      )}
      {simulated ? "Simulated" : "Live"}
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "connected"
      ? "success"
      : status === "error"
        ? "destructive"
        : status === "simulated"
          ? "secondary"
          : "outline";
  return <Badge variant={variant}>{status}</Badge>;
}

function ProtocolCard({ p, canOperate }: { p: ProtocolInfo; canOperate: boolean }) {
  const qc = useQueryClient();
  const onDone = (verb: string) => ({
    onSuccess: (r: { status: string; message: string }) => {
      toast.success(`${p.name}: ${r.message || r.status}`);
    },
    onError: (err: unknown) => {
      toast.error(`${verb} ${p.name} failed`, {
        description: apiErrorMessage(err, "The adapter did not respond."),
      });
    },
    onSettled: () => void qc.invalidateQueries({ queryKey: KEY }),
  });
  const connect = useMutation({ mutationFn: () => connectProtocol(p.name), ...onDone("Connect") });
  const disconnect = useMutation({
    mutationFn: () => disconnectProtocol(p.name),
    ...onDone("Disconnect"),
  });
  const running = OPERATIONAL_STATUSES.has(p.status);
  const busy = connect.isPending || disconnect.isPending;

  return (
    <Card data-testid={`protocol-${p.name}`}>
      <CardHeader className="flex-row items-start justify-between gap-2 space-y-0 pb-2">
        <div>
          <CardTitle className="text-base text-foreground">{p.name}</CardTitle>
          <p className="text-xs text-muted-foreground">version {p.version}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <ModeBadge mode={p.mode} />
          <StatusBadge status={p.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
          <dt className="text-muted-foreground">Messages in</dt>
          <dd className="text-right tabular-nums">{p.messages_received.toLocaleString()}</dd>
          <dt className="text-muted-foreground">Messages out</dt>
          <dd className="text-right tabular-nums">{p.messages_sent.toLocaleString()}</dd>
          <dt className="text-muted-foreground">Errors</dt>
          <dd className={cn("text-right tabular-nums", p.errors > 0 && "text-destructive")}>
            {p.errors.toLocaleString()}
          </dd>
          <dt className="text-muted-foreground">Uptime</dt>
          <dd className="text-right tabular-nums">{formatUptime(p.uptime_seconds)}</dd>
        </dl>
        {canOperate && (
          <div className="flex gap-2">
            {running ? (
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() => disconnect.mutate()}
                aria-label={`Disconnect ${p.name}`}
              >
                <Plug className="h-4 w-4" aria-hidden="true" />
                {disconnect.isPending ? "Disconnecting…" : "Disconnect"}
              </Button>
            ) : (
              <Button
                type="button"
                size="sm"
                disabled={busy}
                onClick={() => connect.mutate()}
                aria-label={`Connect ${p.name}`}
                data-testid={`connect-${p.name}`}
              >
                <PlugZap className="h-4 w-4" aria-hidden="true" />
                {connect.isPending ? "Connecting…" : "Connect"}
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
