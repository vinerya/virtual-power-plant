// Protocol adapters (src/vpp/api/routes/protocols.py).
//
// `mode` is "live" when an adapter talks to a real endpoint and "simulated"
// when it only runs in memory (no external traffic). A simulated adapter's
// status is "simulated", never "connected".
import { z } from "zod";
import { api } from "./client";
import { parseResponse } from "./errors";
import { withMockFallback } from "./mocks";

export const protocolSchema = z.object({
  name: z.string(),
  version: z.string(),
  status: z.string(),
  mode: z.string().default("live"),
  simulated: z.boolean().default(false),
  messages_sent: z.number().default(0),
  messages_received: z.number().default(0),
  errors: z.number().default(0),
  uptime_seconds: z.number().default(0),
});
export type ProtocolInfo = z.infer<typeof protocolSchema>;

export const connectResponseSchema = z.object({
  name: z.string(),
  status: z.string(),
  message: z.string().default(""),
});
export type ConnectResponse = z.infer<typeof connectResponseSchema>;

export function listProtocols(): Promise<ProtocolInfo[]> {
  return withMockFallback(
    async () =>
      parseResponse(z.array(protocolSchema), await api.get("/api/v1/protocols/"), "protocols"),
    () => [
      {
        name: "modbus",
        version: "tcp",
        status: "simulated",
        mode: "simulated",
        simulated: true,
        messages_sent: 0,
        messages_received: 42,
        errors: 0,
        uptime_seconds: 3600,
      },
    ],
  );
}

export async function connectProtocol(name: string): Promise<ConnectResponse> {
  return parseResponse(
    connectResponseSchema,
    await api.post(`/api/v1/protocols/${encodeURIComponent(name)}/connect`, {}),
    "connect",
  );
}

export async function disconnectProtocol(name: string): Promise<ConnectResponse> {
  return parseResponse(
    connectResponseSchema,
    await api.post(`/api/v1/protocols/${encodeURIComponent(name)}/disconnect`),
    "disconnect",
  );
}

/** Statuses in which the adapter is running (connect makes no sense). */
export const OPERATIONAL_STATUSES = new Set(["connected", "simulated"]);

// ---------------------------------------------------------------------------
// Protocol messages: OpenADR DR events, the DR decision log and status
// (src/vpp/api/routes/protocol_ops.py).
// ---------------------------------------------------------------------------

const nullableNumber = z.number().nullable().optional();

export const openadrIntervalSchema = z.object({
  uid: z.string().nullable().optional(),
  start_time: z.number(),
  duration_seconds: z.number(),
  value: z.number().nullable().optional(),
});

export const openadrSignalSchema = z.object({
  signal_name: z.string(),
  signal_type: z.string(),
  signal_id: z.string().nullable().optional(),
  current_value: nullableNumber,
  intervals: z.array(openadrIntervalSchema).default([]),
});

export const openadrEventSchema = z.object({
  event_id: z.string(),
  signal_type: z.string(),
  signal_level: z.number().default(0),
  start_time: z.number(),
  duration_seconds: z.number(),
  end_time: z.number().optional(),
  status: z.string(),
  active: z.boolean().default(false),
  current_value: nullableNumber,
  opt_type: z.string().nullable().optional(),
  opted_at: nullableNumber,
  source: z.string().optional(),
  market_context: z.string().optional(),
  modification_number: z.number().optional(),
  priority: z.number().optional(),
  test_event: z.boolean().optional(),
  response_required: z.string().optional(),
  vtn_status: z.string().nullable().optional(),
  signals: z.array(openadrSignalSchema).default([]),
});
export type OpenADREvent = z.infer<typeof openadrEventSchema>;

export const openadrEventsSchema = z.object({
  protocol: z.string(),
  status: z.string(),
  mode: z.string(),
  simulated: z.boolean().default(false),
  ven_id: z.string().nullable().optional(),
  vtn_id: z.string().nullable().optional(),
  registered: z.boolean().default(false),
  events: z.array(openadrEventSchema).default([]),
});
export type OpenADREvents = z.infer<typeof openadrEventsSchema>;

export async function listOpenADREvents(): Promise<OpenADREvents> {
  return parseResponse(
    openadrEventsSchema,
    await api.get("/api/v1/protocols/openadr/events"),
    "openadr events",
  );
}

export const optOverrideSchema = z.object({
  event_id: z.string(),
  opt_type: z.string(),
  sent_to_vtn: z.boolean().default(false),
  released_dispatch: z.boolean().default(false),
});
export type OptOverride = z.infer<typeof optOverrideSchema>;

export async function overrideOpenADROpt(
  eventId: string,
  optType: "optIn" | "optOut",
  reason = "",
): Promise<OptOverride> {
  return parseResponse(
    optOverrideSchema,
    await api.post(`/api/v1/protocols/openadr/events/${encodeURIComponent(eventId)}/opt`, {
      opt_type: optType,
      reason,
    }),
    "opt override",
  );
}

export const drResponseSchema = z.object({
  id: z.union([z.string(), z.number()]),
  created_at: z.number().nullable(),
  protocol: z.string(),
  source_id: z.string(),
  revision: z.number().default(0),
  action: z.string(),
  opt_type: z.string().nullable().optional(),
  signal_type: z.string().nullable().optional(),
  signal_level: nullableNumber,
  target_kw: nullableNumber,
  delivered_kw: nullableNumber,
  window_start: nullableNumber,
  window_end: nullableNumber,
  run_id: z.string().nullable().optional(),
  reason: z.string().default(""),
  details: z.record(z.unknown()).default({}),
});
export type DRResponse = z.infer<typeof drResponseSchema>;

export async function listDRResponses(limit = 50): Promise<DRResponse[]> {
  return parseResponse(
    z.array(drResponseSchema),
    await api.get(`/api/v1/dr/responses?limit=${limit}`),
    "dr responses",
  );
}

export const drStatusSchema = z.object({
  running: z.boolean(),
  auto_response_enabled: z.boolean(),
  active: z.unknown().nullable().optional(),
  dispatch_count: z.number().default(0),
  last_tick_at: nullableNumber,
  last_error: z.string().nullable().optional(),
  protocols: z.record(z.boolean()).default({}),
  device_control_enabled: z.boolean().default(false),
});
export type DRStatus = z.infer<typeof drStatusSchema>;

export async function getDRStatus(): Promise<DRStatus> {
  return parseResponse(drStatusSchema, await api.get("/api/v1/dr/status"), "dr status");
}

/** Adapters whose traffic the protocol messages panel can show. */
export const DR_PROTOCOLS = new Set(["openadr", "ieee2030_5"]);
