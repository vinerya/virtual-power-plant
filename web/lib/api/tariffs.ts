import { api } from "./client";
import type { Bill } from "./types";

export interface TariffComponent {
  name: string;
  kind: string;
  unit?: string;
  rate?: number | null;
  rates?: number[] | null;
}

export interface Tariff {
  id: string;
  name: string;
  utility?: string | null;
  sector?: string | null;
  source?: string | null;
  components: TariffComponent[];
  tou_heatmap?: number[][];
  metadata?: Record<string, unknown>;
}

export interface SimulateRequest {
  synthetic?: boolean;
  period_days?: number;
  csv?: string;
  compare_to?: string;
}

export interface SimulateResponse {
  bill: Bill;
  comparison?: Bill | null;
}

export function listTariffs(): Promise<Tariff[]> {
  return api.get<Tariff[]>("/api/v1/tariffs");
}

export function getTariff(id: string): Promise<Tariff> {
  return api.get<Tariff>(`/api/v1/tariffs/${encodeURIComponent(id)}`);
}

export function simulateBill(
  id: string,
  req: SimulateRequest,
): Promise<SimulateResponse> {
  return api.post<SimulateResponse>(
    `/api/v1/tariffs/${encodeURIComponent(id)}/simulate`,
    req,
  );
}
