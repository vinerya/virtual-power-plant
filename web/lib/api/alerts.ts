import { api } from "./client";
import type { Alert, AlertSeverity } from "./types";

export interface ListAlertsParams {
  since?: string; // ISO timestamp
  severity?: AlertSeverity | "all";
  status?: "active" | "acknowledged" | "snoozed" | "all";
  limit?: number;
}

export async function listAlerts(
  params: ListAlertsParams = {},
): Promise<Alert[]> {
  const q = new URLSearchParams();
  if (params.since) q.set("since", params.since);
  if (params.severity && params.severity !== "all")
    q.set("severity", params.severity);
  if (params.status && params.status !== "all") q.set("status", params.status);
  if (params.limit != null) q.set("limit", String(params.limit));
  const qs = q.toString();
  try {
    return await api.get<Alert[]>(`/api/v1/alerts${qs ? `?${qs}` : ""}`);
  } catch (e) {
    // Backend may not have implemented alerts yet — return demo data so the
    // page renders. Documented in README.
    if ((e as { status?: number })?.status === 404) {
      return demoAlerts();
    }
    throw e;
  }
}

export function ackAlert(id: string): Promise<Alert> {
  return api.post<Alert>(
    `/api/v1/alerts/${encodeURIComponent(id)}/ack`,
    {},
  );
}

export function snoozeAlert(
  id: string,
  durationMs: number,
): Promise<Alert> {
  const until = new Date(Date.now() + durationMs).toISOString();
  return api.post<Alert>(
    `/api/v1/alerts/${encodeURIComponent(id)}/snooze`,
    { until, duration_ms: durationMs },
  );
}

export async function bulkAckAlerts(ids: string[]): Promise<void> {
  await Promise.all(ids.map((id) => ackAlert(id).catch(() => null)));
}

// Used as a fallback when the backend does not implement /alerts yet.
function demoAlerts(): Alert[] {
  const now = Date.now();
  const mk = (
    i: number,
    sev: AlertSeverity,
    title: string,
    message: string,
    src: string,
    sourceLink?: string,
    status: Alert["status"] = "active",
  ): Alert => ({
    id: `demo-${i}`,
    timestamp: new Date(now - i * 1000 * 60 * 30).toISOString(),
    severity: sev,
    source: src,
    source_kind: src.startsWith("res-") ? "resource" : "system",
    source_link: sourceLink,
    title,
    message,
    status,
  });
  return [
    mk(
      1,
      "critical",
      "Battery offline",
      "Heartbeat missing for 5m on res-bat-001",
      "res-bat-001",
      "/assets/res-bat-001",
    ),
    mk(
      2,
      "warning",
      "SOH degraded",
      "State-of-health dropped below 0.85 on res-bat-007",
      "res-bat-007",
      "/assets/res-bat-007",
    ),
    mk(
      3,
      "info",
      "Tariff updated",
      "TOU schedule for PG&E E-19 was edited",
      "tariff:pge-e-19",
      "/tariffs/pge-e-19",
    ),
    mk(
      4,
      "warning",
      "Solver fallback",
      "MILP solver fell back to LP relaxation on dispatch run",
      "system",
      "/trading/dispatches",
    ),
    mk(
      5,
      "info",
      "New resource registered",
      "res-pv-104 came online",
      "res-pv-104",
      "/assets/res-pv-104",
    ),
    mk(
      6,
      "critical",
      "Inverter fault",
      "Inverter F1 fault on res-pv-077",
      "res-pv-077",
      "/assets/res-pv-077",
      "acknowledged",
    ),
  ];
}
