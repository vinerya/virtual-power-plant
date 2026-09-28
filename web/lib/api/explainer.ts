import { api } from "./client";
import type { ExplainerResponse } from "./types";

/**
 * Explainer for a dispatch run. Resolves to `null` when the backend has no
 * explanation for the run (404) so the UI can show an empty state; other
 * failures reject and are rendered as errors.
 */
export async function getExplainer(
  runId: string,
): Promise<ExplainerResponse | null> {
  try {
    return await api.get<ExplainerResponse>(
      `/api/v1/dispatches/${encodeURIComponent(runId)}/explain`,
    );
  } catch (e) {
    if ((e as { status?: number })?.status === 404) return null;
    throw e;
  }
}
