import { api } from "./client";

export interface ConfigDocument {
  yaml: string;
  updated_at?: string;
  hash?: string;
}

export function getConfig(): Promise<ConfigDocument> {
  return api.get<ConfigDocument>("/api/v1/config");
}

export function getConfigSchema(): Promise<Record<string, unknown> | null> {
  return api.get<Record<string, unknown>>("/api/v1/config/schema");
}

export function applyConfig(yaml: string): Promise<ConfigDocument> {
  return api.put<ConfigDocument>("/api/v1/config", { yaml });
}
