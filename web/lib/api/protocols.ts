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
