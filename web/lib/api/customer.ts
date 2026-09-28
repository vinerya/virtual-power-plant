import { api } from "./client";
import { withMockFallback as withFallback } from "./mocks";
import type {
  Bill,
  Customer,
  CustomerDevice,
  DRProgram,
} from "./types";

export interface CustomerBillResponse {
  bill: Bill;
  baseline?: Bill | null;
  savings?: number;
  this_month_kwh?: number;
  last_month_kwh?: number;
}

export interface CustomerEnrollmentRequest {
  program_ids: string[];
  acknowledged: boolean;
}

// Every call below falls back to demo data ONLY when mock mode is enabled
// (NEXT_PUBLIC_USE_MOCKS=1, see ./mocks.ts). Otherwise errors propagate.

export function getMe(): Promise<Customer> {
  return withFallback(
    () => api.get<Customer>("/api/v1/customer/me"),
    () => ({
      id: "demo-cust",
      name: "Demo Household",
      address: "123 Sunlit Lane, Austin TX",
      tariff_id: "flat-residential",
      baseline_kwh_per_month: 850,
      email: "demo@example.com",
    }),
  );
}

export function getMyBill(month?: string): Promise<CustomerBillResponse> {
  const qs = month ? `?month=${encodeURIComponent(month)}` : "";
  return withFallback(
    () => api.get<CustomerBillResponse>(`/api/v1/customer/me/bill${qs}`),
    () => demoBill(),
  );
}

export function getMyDevices(): Promise<CustomerDevice[]> {
  return withFallback(
    () => api.get<CustomerDevice[]>("/api/v1/customer/me/devices"),
    () => [
      {
        id: "bat-1",
        kind: "battery",
        name: "Powerwall",
        state: "discharging",
        current_power: -2.4,
        state_of_charge: 0.62,
      },
      {
        id: "ev-1",
        kind: "ev",
        name: "Model 3",
        state: "idle",
        current_power: 0,
        state_of_charge: 0.78,
      },
      {
        id: "th-1",
        kind: "thermostat",
        name: "Living room",
        state: "cooling",
        setpoint_c: 22,
      },
    ],
  );
}

export function listPrograms(): Promise<DRProgram[]> {
  return withFallback(
    () => api.get<DRProgram[]>("/api/v1/customer/programs"),
    () => [
      {
        id: "summer-peak",
        name: "Summer Peak Saver",
        description:
          "Earn $1.25/kWh reduced during summer peak events (3pm–7pm).",
        utility: "Generic",
        incentive_per_event: 25,
      },
      {
        id: "winter-balance",
        name: "Winter Balance",
        description:
          "Pre-heat your home and sell stored energy to the grid in cold snaps.",
        utility: "Generic",
        incentive_per_event: 18,
      },
      {
        id: "always-on",
        name: "Always-on Battery",
        description:
          "Allow VPP to dispatch your battery for grid services year-round.",
        utility: "Generic",
        incentive_per_event: 12,
      },
    ],
  );
}

export function postEnrollment(
  req: CustomerEnrollmentRequest,
): Promise<{ ok: true; enrolled: string[] }> {
  return withFallback(
    () => api.post("/api/v1/customer/enrollments", req),
    () => ({ ok: true, enrolled: req.program_ids }),
  );
}

function demoBill(): CustomerBillResponse {
  const bill: Bill = {
    total: 142.7,
    currency: "USD",
    line_items: [
      { kind: "fixed", name: "Customer charge", amount: 12 },
      { kind: "energy", name: "Energy (820 kWh)", amount: 130.7 },
    ],
    period: { start: "2026-04-01", end: "2026-04-30" },
  };
  const baseline: Bill = {
    total: 178.4,
    currency: "USD",
    line_items: [
      { kind: "fixed", name: "Customer charge", amount: 12 },
      { kind: "energy", name: "Energy (1020 kWh)", amount: 166.4 },
    ],
    period: { start: "2026-04-01", end: "2026-04-30" },
  };
  return {
    bill,
    baseline,
    savings: baseline.total - bill.total,
    this_month_kwh: 820,
    last_month_kwh: 905,
  };
}
