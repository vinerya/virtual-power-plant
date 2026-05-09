import { api } from "./client";
import type { ExplainerResponse } from "./types";

export function getExplainer(
  runId: string,
): Promise<ExplainerResponse | null> {
  return api.get<ExplainerResponse>(
    `/api/v1/dispatches/${encodeURIComponent(runId)}/explain`,
  );
}
