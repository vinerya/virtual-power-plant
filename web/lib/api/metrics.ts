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
    // Backend may not implement the metrics endpoint yet; the asset page
    // falls back to a client-side ring buffer.
    return null;
  }
}
