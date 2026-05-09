import { api } from "./client";
import type { ResourceResponse } from "./types";

export function listResources(): Promise<ResourceResponse[]> {
  return api.get<ResourceResponse[]>("/api/v1/resources");
}

export function getResource(id: string): Promise<ResourceResponse> {
  return api.get<ResourceResponse>(`/api/v1/resources/${encodeURIComponent(id)}`);
}
