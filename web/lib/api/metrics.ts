import { api } from "./client";
import type { ResourceMetricsResponse } from "./types";

export async function getResourceMetrics(
  id: string,
  window: "1h" | "24h" | "7d" = "24h",
): Promise<ResourceMetricsResponse | null> {
  try {
    return await api.get<ResourceMetricsResponse>(
      `/api/v1/resources/${encodeURIComponent(id)}/metrics?window=${window}`,
    );
  } catch (e) {
    // A 404 means this backend has no stored history for the resource; the
    // asset page then builds a client-side ring buffer from live polls.
    // Anything else is a real failure and is surfaced to the caller.
    if ((e as { status?: number })?.status === 404) return null;
    throw e;
  }
}
