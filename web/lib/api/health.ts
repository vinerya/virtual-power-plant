import { api } from "./client";
import type { HealthResponse } from "./types";

export function getHealth(): Promise<HealthResponse> {
  return api.get<HealthResponse>("/health");
}
