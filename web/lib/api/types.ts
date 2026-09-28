// Shared API types used across the operator console.
// Backend lives in `src/...` (FastAPI). These mirror the response shapes.

export interface Token {
  access_token: string;
  token_type: string;
  expires_at?: string;
  expires_in?: number;
}

export interface ResourceResponse {
  id: string;
  name: string;
  resource_type: "battery" | "solar" | "wind" | string;
  rated_power: number;
  current_power: number;
  online: boolean;
  efficiency?: number | null;
  state_of_charge?: number | null;
  state_of_health?: number | null;
  capacity_kwh?: number | null;
  cycle_count?: number | null;
  charge_limit_kw?: number | null;
  discharge_limit_kw?: number | null;
  irradiance?: number | null;
  dc_capacity_kw?: number | null;
  ac_capacity_kw?: number | null;
  wind_speed_ms?: number | null;
  cut_in_speed?: number | null;
  cut_out_speed?: number | null;
  created_at?: string;
  updated_at: string;
  metadata?: Record<string, unknown> | null;
  // Allow arbitrary extra subtype-specific fields.
  [key: string]: unknown;
}

export interface ResourceMetricsPoint {
  timestamp: string;
  power: number;
  state_of_charge?: number;
  efficiency?: number;
}

export interface ResourceMetricsResponse {
  resource_id: string;
  window: string;
  points: ResourceMetricsPoint[];
}

export interface DispatchRun {
  id: string;
  resource_id?: string;
  status: string;
  started_at?: string;
  finished_at?: string | null;
  created_at: string;
  problem_type: string;
  fallback_used?: boolean;
  objective_value?: number | null;
  solver?: string | null;
  iterations?: number | null;
  gap?: number | null;
  solve_time_ms?: number | null;
  inputs?: Record<string, unknown>;
  solution?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  total_cost?: number | null;
}

export interface ExplainerRunStep {
  step: number;
  charge?: number;
  discharge?: number;
  power?: number;
  price?: number;
}

export interface ExplainerRun {
  name: string;
  total_cost: number;
  per_step: ExplainerRunStep[];
}

export interface ExplainerBindingConstraint {
  name: string;
  reason?: string;
  description?: string;
  step?: number;
  slack: number;
}

export interface ExplainerResponse {
  run_id: string;
  actual: ExplainerRun;
  counterfactuals: ExplainerRun[];
  binding_constraints: ExplainerBindingConstraint[];
  rationale?: string;
}

export interface BillLineItem {
  kind:
    | "fixed"
    | "energy"
    | "tier"
    | "demand"
    | "minimum"
    | "adder"
    | "tax"
    | "credit"
    | string;
  name: string;
  amount: number;
  quantity?: number | null;
  unit?: string | null;
  rate?: number | null;
}

export interface Bill {
  total: number;
  currency?: string;
  tariff_id?: string;
  line_items: BillLineItem[];
  period?: { start: string; end: string };
  metadata?: Record<string, unknown>;
}

export interface ConfigValidationError {
  path: string;
  message: string;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "down" | string;
  version?: string;
  uptime_s?: number;
}

// ----- Alerts (M4) -----
export type AlertSeverity = "info" | "warning" | "critical";
export type AlertStatus = "active" | "acknowledged" | "snoozed";

export interface Alert {
  id: string;
  timestamp: string;
  severity: AlertSeverity;
  source: string; // resource id, "system", tariff:..., etc.
  source_kind?: "resource" | "tariff" | "system" | "dispatch" | string;
  source_link?: string; // optional canonical URL on the operator console
  title: string;
  message: string;
  status: AlertStatus;
  snoozed_until?: string | null;
  acknowledged_at?: string | null;
}

// ----- Sites (M4) -----
export interface Site {
  id: string;
  name: string;
  lat: number;
  lon: number;
  region?: string;
  resource_ids: string[];
  total_resources: number;
  online_count: number;
  current_power: number;
  rated_power: number;
  active_alerts: number;
  health: "green" | "yellow" | "red";
}

// ----- Customer portal (M4) -----
export interface Customer {
  id: string;
  name: string;
  address?: string;
  tariff_id?: string;
  baseline_kwh_per_month?: number;
  email?: string;
}

export interface CustomerDevice {
  id: string;
  kind: "battery" | "ev" | "thermostat" | string;
  name: string;
  state: string;
  current_power?: number;
  state_of_charge?: number;
  setpoint_c?: number;
}

export interface DRProgram {
  id: string;
  name: string;
  description: string;
  utility?: string;
  incentive_per_event?: number;
  enrolled?: boolean;
}
