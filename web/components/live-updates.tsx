"use client";

/**
 * Mounts the singleton WebSocket client at the operator-shell level.
 *
 * - `resource_updates`    → invalidate ["resources"] and ["resource", id]
 * - `optimization_events` → invalidate ["dispatches"]
 * - `alerts`              → invalidate ["alerts"] + toast via sonner
 *
 * Broadcasts bridged from the backend EventBus carry a BusEvent in
 * `msg.data` (`{event_type, data, source, severity, ...}`); other
 * broadcasters may send a flat object. Both are handled.
 */

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { asBusEvent, getWsClient, type WsStatus } from "@/lib/ws/client";

const CHANNELS = ["resource_updates", "optimization_events", "alerts"];

function humanize(s: string): string {
  const t = s.replace(/[_.]+/g, " ").trim();
  return t.charAt(0).toUpperCase() + t.slice(1);
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

export function LiveUpdates() {
  const qc = useQueryClient();

  useEffect(() => {
    const client = getWsClient();
    client.start({ channels: CHANNELS });
    const off = client.on((msg) => {
      const ev = asBusEvent(msg.data);
      const flat = (ev ? ev.data : (msg.data ?? {})) as Record<string, unknown>;

      switch (msg.channel) {
        case "resource_updates": {
          qc.invalidateQueries({ queryKey: ["resources"] });
          qc.invalidateQueries({ queryKey: ["sites"] });
          const id = str(flat.resource_id) ?? str(flat.id);
          if (id) qc.invalidateQueries({ queryKey: ["resource", id] });
          break;
        }
        case "optimization_events":
          qc.invalidateQueries({ queryKey: ["dispatches"] });
          break;
        case "alerts": {
          qc.invalidateQueries({ queryKey: ["alerts"] });
          const severity = str(flat.severity) ?? ev?.severity;
          const title =
            str(flat.title) ?? (ev ? humanize(ev.event_type) : "Alert");
          const desc = str(flat.message) ?? str(ev?.source) ?? "";
          if (severity === "critical" || severity === "error") {
            toast.error(title, { description: desc });
          } else if (severity === "warning") {
            toast.warning(title, { description: desc });
          } else {
            toast(title, { description: desc });
          }
          break;
        }
        default:
          break;
      }
    });
    return () => {
      off();
      client.stop();
    };
  }, [qc]);

  return null;
}

const STATUS_LABEL: Record<WsStatus, string> = {
  idle: "live: off",
  connecting: "live: connecting…",
  open: "live",
  reconnecting: "live: reconnecting…",
  unauthorized: "live: signed out",
  disabled: "live: demo mode",
};

/** Small badge showing the real-time connection state. */
export function LiveStatusBadge() {
  const [status, setStatus] = useState<WsStatus>("idle");
  useEffect(() => getWsClient().onStatus(setStatus), []);
  const variant =
    status === "open"
      ? "success"
      : status === "unauthorized"
        ? "destructive"
        : "outline";
  return (
    <Badge
      variant={variant}
      aria-live="polite"
      title="Real-time event stream (WebSocket)"
      data-testid="live-status"
    >
      {STATUS_LABEL[status]}
    </Badge>
  );
}
