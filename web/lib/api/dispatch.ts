import { api } from "./client";
import type { DispatchRun } from "./types";

export interface ListDispatchesParams {
  start?: string;
  end?: string;
  resource_ids?: string[];
  limit?: number;
  offset?: number;
}

export function listDispatches(
  params: ListDispatchesParams = {},
): Promise<DispatchRun[]> {
  const q = new URLSearchParams();
  if (params.start) q.set("start", params.start);
  if (params.end) q.set("end", params.end);
  if (params.limit != null) q.set("limit", String(params.limit));
  if (params.offset != null) q.set("offset", String(params.offset));
  if (params.resource_ids && params.resource_ids.length) {
    for (const id of params.resource_ids) q.append("resource_id", id);
  }
  const qs = q.toString();
  return api.get<DispatchRun[]>(
    `/api/v1/optimization/history${qs ? `?${qs}` : ""}`,
  );
}
