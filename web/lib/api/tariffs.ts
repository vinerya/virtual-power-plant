// Tariff API — mirrors src/vpp/schemas/tariffs.py exactly.
//
// `urdb_json` is the stored source of truth; everything after it on
// `Tariff` is derived server-side from the same parsed components the bill
// engine uses, so the UI never re-implements URDB parsing.

import { api } from "./client";
import type { Bill } from "./types";

export interface TariffTier {
  max_kwh: number | null;
  rate: number;
}

export interface TariffComponent {
  name: string;
  /** energy | tier | demand | fixed | minimum | adder | tax */
  kind: string;
  /** $/kWh, $/kW, $/month, $/day, or "fraction" (0.03 = 3%) */
  unit: string;
  rate?: number | null;
  rates?: number[] | null;
  tiers?: TariffTier[] | null;
  sell_rate?: number | null;
  schedule?: string[];
  detail?: string | null;
}

export type NemRegime = "none" | "nem2" | "nem3" | "net_billing";

export interface Tariff {
  id: string;
  name: string;
  utility: string;
  urdb_label?: string | null;
  urdb_json: Record<string, unknown>;
  effective_date?: string | null;
  created_at: string;
  updated_at: string;
  // derived
  sector?: string | null;
  source?: string | null;
  description?: string | null;
  components: TariffComponent[];
  /** 12 months x 24 hours, weekday import $/kWh */
  tou_heatmap?: number[][] | null;
  tou_heatmap_weekend?: number[][] | null;
  is_tou: boolean;
  nem_regime: NemRegime | string;
  nem_source: string;
  parse_error?: string | null;
}

export interface TariffWrite {
  name: string;
  utility?: string;
  urdb_json: Record<string, unknown>;
  effective_date?: string | null;
  urdb_label?: string | null;
}

export interface TariffPresetSummary {
  id: string;
  name: string;
  utility?: string | null;
  sector?: string | null;
  description?: string | null;
  source_date?: string | null;
  illustrative: boolean;
}

export interface TariffPreset extends TariffPresetSummary {
  urdb_json: Record<string, unknown>;
}

export interface URDBImportStatus {
  configured: boolean;
  detail: string;
}

export interface SyntheticLoadSpec {
  profile?: "residential" | "commercial";
  avg_kw?: number;
  pv_kw?: number;
  interval_minutes?: 15 | 30 | 60;
}

export interface SimulateRequest {
  // exactly one load source
  synthetic?: boolean | SyntheticLoadSpec;
  csv?: string;
  period_days?: number;
  billing_period_start?: string;
  billing_period_end?: string;
  timezone?: string;
  billing_cycle?: "auto" | "single" | "monthly";
  /** Omit to use the tariff's own regime. */
  nem?: NemRegime;
  nem3_avoided_cost?: number[];
  compare_to?: string;
}

export interface BillLineItemDTO {
  kind: string;
  label: string;
  quantity: number;
  unit: string;
  rate: number;
  amount: number;
}

export interface BillCycle {
  period_start: string;
  period_end: string;
  total: number;
  export_credit: number;
}

export interface LoadSummary {
  source: "meter_trace" | "synthetic" | "csv";
  method?: string | null;
  timezone: string;
  interval_minutes: number;
  intervals: number;
  import_kwh: number;
  export_kwh: number;
  peak_kw: number;
}

export interface BillSimulation {
  total: number;
  tariff_name: string;
  tariff_id?: string | null;
  currency: string;
  line_items: BillLineItemDTO[];
  period_start: string;
  period_end: string;
  cycles: BillCycle[];
  nem_regime: string;
  nem_source: string;
  export_kwh: number;
  export_credit: number;
  notes: string[];
  load_summary?: LoadSummary | null;
  comparison?: BillSimulation | null;
}

const BASE = "/api/v1/tariffs";
const enc = encodeURIComponent;

export function listTariffs(): Promise<Tariff[]> {
  return api.get<Tariff[]>(`${BASE}?limit=200`);
}

export function getTariff(id: string): Promise<Tariff> {
  return api.get<Tariff>(`${BASE}/${enc(id)}`);
}

export function createTariff(body: TariffWrite): Promise<Tariff> {
  return api.post<Tariff>(BASE, body);
}

export function updateTariff(id: string, body: Partial<TariffWrite>): Promise<Tariff> {
  return api.put<Tariff>(`${BASE}/${enc(id)}`, body);
}

export function deleteTariff(id: string): Promise<void> {
  return api.delete<void>(`${BASE}/${enc(id)}`);
}

export function listTariffPresets(): Promise<TariffPresetSummary[]> {
  return api.get<TariffPresetSummary[]>(`${BASE}/presets`);
}

export function getTariffPreset(id: string): Promise<TariffPreset> {
  return api.get<TariffPreset>(`${BASE}/presets/${enc(id)}`);
}

export function getUrdbImportStatus(): Promise<URDBImportStatus> {
  return api.get<URDBImportStatus>(`${BASE}/import-urdb`);
}

export function importUrdb(urdb_label: string, name_override?: string): Promise<Tariff> {
  return api.post<Tariff>(`${BASE}/import-urdb`, {
    urdb_label,
    name_override: name_override || undefined,
  });
}

export function simulateBill(id: string, req: SimulateRequest): Promise<BillSimulation> {
  return api.post<BillSimulation>(`${BASE}/${enc(id)}/simulate`, req);
}

/** Adapt a simulation result to the presentational `Bill` used by BillBreakdown. */
export function simulationToBill(sim: BillSimulation): Bill {
  return {
    total: sim.total,
    currency: sim.currency,
    tariff_id: sim.tariff_id ?? undefined,
    line_items: sim.line_items.map((li) => ({
      kind: li.kind,
      name: li.label,
      amount: li.amount,
      quantity: li.quantity,
      unit: li.unit,
      rate: li.rate,
    })),
    period: { start: sim.period_start, end: sim.period_end },
    metadata: { tariff_name: sim.tariff_name },
  };
}

/** FastAPI error detail (string or validation list) → readable message. */
export function apiErrorMessage(e: unknown, fallback: string): string {
  if (!e || typeof e !== "object") return fallback;
  const status = (e as { status?: number }).status;
  const detail = (e as { detail?: unknown }).detail;
  let msg: string | null = null;
  if (detail && typeof detail === "object" && "detail" in detail) {
    const d = (detail as { detail: unknown }).detail;
    if (typeof d === "string") msg = d;
    else if (Array.isArray(d))
      msg = d
        .map((x) => (x && typeof x === "object" && "msg" in x ? String(x.msg) : String(x)))
        .join("; ");
  } else if (typeof detail === "string" && detail) {
    msg = detail;
  }
  if (status === 403 && !msg) msg = "This action requires the admin role.";
  return msg ?? (status ? `${fallback} (HTTP ${status})` : fallback);
}
