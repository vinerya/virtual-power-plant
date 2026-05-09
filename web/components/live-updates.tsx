"use client";

/**
 * Mounts the singleton WebSocket client at the operator-shell level.
 *
 * - `resource_updates`  → invalidate ["resources"] and ["resource", id]
 * - `optimization_events` → invalidate ["dispatches"]
 * - `alerts` → toast via sonner
 */

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { getWsClient } from "@/lib/ws/client";

export function LiveUpdates() {
  const qc = useQueryClient();

  useEffect(() => {
    const client = getWsClient();
    client.start({
      channels: ["resource_updates", "optimization_events", "alerts"],
    });
    const off = client.on((msg) => {
      switch (msg.channel) {
        case "resource_updates": {
          qc.invalidateQueries({ queryKey: ["resources"] });
          const data = msg.data as { id?: string; resource_id?: string } | null;
          const id = data?.id || data?.resource_id;
          if (id) qc.invalidateQueries({ queryKey: ["resource", id] });
          break;
        }
        case "optimization_events":
          qc.invalidateQueries({ queryKey: ["dispatches"] });
          break;
        case "alerts": {
          const data = msg.data as
            | { severity?: string; title?: string; message?: string }
            | null;
          const title = data?.title || "Alert";
          const desc = data?.message || "";
          if (data?.severity === "critical" || data?.severity === "error") {
            toast.error(title, { description: desc });
          } else if (data?.severity === "warning") {
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
      // Keep the singleton alive across route changes — only stop on unmount
      // of the operator shell, which is the page lifetime in practice.
      client.stop();
    };
  }, [qc]);

  return null;
}
