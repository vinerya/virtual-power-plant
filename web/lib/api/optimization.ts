// Horizon scheduling and closed-loop MPC backtests
// (src/vpp/api/routes/optimization.py, src/vpp/schemas/optimization.py).
//
// Prices are currency per kWh. Power is kW with `power = charge - discharge`
// (positive = charging). Every call persists an optimization run, whose
// `run_id` can be opened in the dispatch explainer (/trading/dispatches?run=).
import { z } from "zod";
import { api } from "./client";
import { parseResponse } from "./errors";

const num = z.number();

export const MAX_SCHEDULE_STEPS = 288;
export const MAX_BACKTEST_STEPS = 672;
/** interval_minutes must divide 60. */
export const INTERVALS = [5, 10, 15, 20, 30, 60] as const;

export interface ScheduleRequest {
  resource_ids?: string[];
  prices?: number[];
  tariff_id?: string;
  horizon_hours?: number;
  horizon_start?: string;
  nem?: "none" | "nem2" | "nem3";
  interval_minutes: number;
  degradation_aware?: boolean;
  replacement_cost_per_kwh?: number;
  terminal_soc_policy?: "value" | "hold";
  feeder_max_import_kw?: number;
  feeder_max_export_kw?: number;
}

export const scheduleResponseSchema = z.object({
  run_id: z.string(),
  status: z.string(),
  method: z.string(),
  fallback_used: z.boolean(),
  fallback_reason: z.string().nullable().optional(),
  solve_time_ms: num,
  objective_value: num.nullable().optional(),
  energy_cost: num,
  wear_cost: num.default(0),
  interval_minutes: num,
  prices: z.array(num),
  charge: z.array(num),
  discharge: z.array(num),
  power: z.array(num),
  per_resource: z.record(
    z.object({ charge: z.array(num), discharge: z.array(num), soc: z.array(num) }),
  ),
  resources: z.array(z.record(z.unknown())),
  tariff_id: z.string().nullable().optional(),
  terminal_soc_policy: z.string(),
  notes: z.array(z.string()).default([]),
});
export type ScheduleResponse = z.infer<typeof scheduleResponseSchema>;

export async function planSchedule(body: ScheduleRequest): Promise<ScheduleResponse> {
  return parseResponse(
    scheduleResponseSchema,
    await api.post("/api/v1/optimization/schedule", body),
    "schedule",
  );
}

export interface BatterySpec {
  capacity_kwh: number;
  max_power_kw: number;
  soc_init?: number;
  soc_min?: number;
  soc_max?: number;
  eta_charge?: number;
  eta_discharge?: number;
  state_of_health?: number;
}

export interface BacktestRequest {
  resource_id?: string;
  battery?: BatterySpec;
  prices: number[];
  interval_minutes: number;
  horizon_steps: number;
  forecast_mode: "perfect" | "persistence" | "noisy";
  noise_sigma?: number;
  seed?: number;
  terminal_soc_policy?: "value" | "hold";
  compare_offline?: boolean;
}

export const backtestResponseSchema = z.object({
  run_id: z.string(),
  status: z.string(),
  ticks: num,
  realized_cost: num,
  realized_cost_adjusted: num,
  no_action_cost: num,
  rules_cost: num,
  rules_cost_adjusted: num,
  perfect_foresight_cost_adjusted: num.nullable().optional(),
  perfect_foresight_status: z.string(),
  regret: num.nullable().optional(),
  terminal_energy_value_per_kwh: num,
  terminal_soc_policy: z.string(),
  final_soc: num,
  fallback_count: num,
  cumulative_solve_time_ms: num,
  cumulative_solver_iterations: num,
  wall_time_s: num,
  forecast_mode: z.string(),
  interval_minutes: num,
  horizon_steps: num,
  charge: z.array(num),
  discharge: z.array(num),
  power: z.array(num),
  soc: z.array(num),
  notes: z.array(z.string()).default([]),
});
export type BacktestResponse = z.infer<typeof backtestResponseSchema>;

export async function runBacktest(body: BacktestRequest): Promise<BacktestResponse> {
  return parseResponse(
    backtestResponseSchema,
    await api.post("/api/v1/optimization/backtest", body),
    "backtest",
  );
}

const tariffOptionSchema = z.object({
  id: z.string(),
  name: z.string(),
  utility: z.string().nullable().optional(),
});
export type TariffOption = z.infer<typeof tariffOptionSchema>;

/** Stored tariffs (id + name only) for the "prices from tariff" picker. */
export async function listTariffOptions(): Promise<TariffOption[]> {
  return parseResponse(
    z.array(tariffOptionSchema),
    await api.get("/api/v1/tariffs?limit=200"),
    "tariffs",
  );
}

/**
 * Parse a pasted price list: numbers separated by commas, whitespace,
 * semicolons or newlines. Returns the numbers and any tokens that were not
 * numbers.
 */
export function parseSeries(text: string): { values: number[]; invalid: string[] } {
  const values: number[] = [];
  const invalid: string[] = [];
  for (const tok of text.split(/[\s,;]+/).filter(Boolean)) {
    const v = Number(tok);
    if (Number.isFinite(v)) values.push(v);
    else invalid.push(tok);
  }
  return { values, invalid };
}

/** A synthetic but plausible day-shaped price curve (currency/kWh) for quick starts. */
export function sampleDayPrices(steps: number, intervalMinutes: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < steps; i++) {
    const h = ((i * intervalMinutes) / 60) % 24;
    const morning = Math.exp(-((h - 8) ** 2) / 4) * 0.08;
    const evening = Math.exp(-((h - 19) ** 2) / 5) * 0.2;
    const solarDip = -Math.exp(-((h - 13) ** 2) / 6) * 0.05;
    out.push(Number((0.12 + morning + evening + solarDip).toFixed(4)));
  }
  return out;
}
