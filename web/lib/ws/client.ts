// Singleton WebSocket client for the FastAPI event stream.
//
// Why it looks like this:
// - Next.js route handlers cannot proxy a WebSocket upgrade, so the browser
//   dials FastAPI directly (`/api/v1/ws`). The URL comes from
//   NEXT_PUBLIC_WS_URL or, at runtime, from the `/api/auth/ws-token` route.
// - Browsers cannot set headers on a WebSocket handshake. Before every
//   (re)connect we fetch a short-lived, socket-only token from
//   `/api/auth/ws-token` (which reads the httpOnly session cookie) and pass
//   it as the `bearer, <token>` subprotocol pair — keeping it out of URLs
//   and access logs.
// - Channels are sent as the `channels` query param (subscribed on
//   connect), so reconnects resubscribe automatically; channels added while
//   open are subscribed with `{"action":"subscribe"}` messages.
//
// Server → client envelope (see src/vpp/api/websocket.py):
//   broadcast: {"channel": "...", "data": {...}, "timestamp": "..."}
//   control:   {"ack": "subscribed:x"} | {"pong": "..."} | {"error": "..."}
// For EventBus-originated broadcasts `data` is a BusEvent (below).

import { USE_MOCKS } from "@/lib/api/mocks";

export type WsChannel =
  | "resource_updates"
  | "optimization_events"
  | "market_data"
  | "alerts"
  | "grid_events"
  | "system"
  | "*";

export interface WsMessage {
  channel: string;
  data: unknown;
  timestamp?: string;
}

/** Shape of `WsMessage.data` for events bridged from the backend EventBus. */
export interface BusEvent {
  event_id: string;
  event_type: string;
  data: Record<string, unknown>;
  source: string;
  severity: string;
  timestamp: number | string;
}

export function asBusEvent(data: unknown): BusEvent | null {
  if (
    data &&
    typeof data === "object" &&
    typeof (data as BusEvent).event_type === "string" &&
    typeof (data as BusEvent).event_id === "string"
  ) {
    const ev = data as BusEvent;
    return { ...ev, data: (ev.data ?? {}) as Record<string, unknown> };
  }
  return null;
}

export type WsStatus =
  | "idle" // not started
  | "connecting"
  | "open"
  | "reconnecting"
  | "unauthorized" // session missing/expired; stopped retrying
  | "disabled"; // mock mode — no live backend

type Handler = (msg: WsMessage) => void;
type StatusHandler = (status: WsStatus) => void;

interface StartOptions {
  channels: string[];
}

interface WsTokenResponse {
  token: string;
  expires_in: number;
  url?: string;
}

const MIN_BACKOFF_MS = 1_000;
const MAX_BACKOFF_MS = 30_000;
const HEARTBEAT_MS = 25_000;
const POLICY_VIOLATION = 1008;

class UnauthorizedError extends Error {}

class WsClient {
  private ws: WebSocket | null = null;
  private handlers = new Set<Handler>();
  private statusHandlers = new Set<StatusHandler>();
  private channels = new Set<string>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private attempt = 0;
  private stopped = true;
  // Bumped on every start/stop so stale async connects are discarded
  // (React StrictMode mounts effects twice in development).
  private generation = 0;
  private _status: WsStatus = "idle";

  get status(): WsStatus {
    return this._status;
  }

  start(opts: StartOptions) {
    for (const c of opts.channels) this.channels.add(c);
    if (!this.stopped) {
      // Already running: subscribe any new channels on the live socket.
      for (const c of opts.channels) this.send({ action: "subscribe", channel: c });
      return;
    }
    this.stopped = false;
    this.generation += 1;
    if (USE_MOCKS) {
      this.setStatus("disabled");
      return;
    }
    this.attempt = 0;
    void this.connect(this.generation);
  }

  stop() {
    this.stopped = true;
    this.generation += 1;
    this.clearTimers();
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      ws.onclose = null;
      try {
        ws.close(1000, "client stop");
      } catch {
        /* noop */
      }
    }
    this.setStatus("idle");
  }

  subscribe(channel: string) {
    if (this.channels.has(channel)) return;
    this.channels.add(channel);
    this.send({ action: "subscribe", channel });
  }

  unsubscribe(channel: string) {
    if (!this.channels.delete(channel)) return;
    this.send({ action: "unsubscribe", channel });
  }

  on(handler: Handler): () => void {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    handler(this._status);
    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  send(msg: unknown) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
  }

  // ------------------------------------------------------------------

  private setStatus(s: WsStatus) {
    if (this._status === s) return;
    this._status = s;
    for (const h of this.statusHandlers) h(s);
  }

  private clearTimers() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private async fetchToken(): Promise<WsTokenResponse> {
    const res = await fetch("/api/auth/ws-token", {
      cache: "no-store",
      credentials: "same-origin",
    });
    if (res.status === 401) throw new UnauthorizedError("session expired");
    if (!res.ok) throw new Error(`ws-token ${res.status}`);
    return (await res.json()) as WsTokenResponse;
  }

  private buildUrl(serverUrl?: string): string {
    const base =
      process.env.NEXT_PUBLIC_WS_URL ||
      serverUrl ||
      `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.hostname}:8000/api/v1/ws`;
    const url = new URL(base, window.location.href);
    if (this.channels.size > 0) {
      url.searchParams.set("channels", Array.from(this.channels).join(","));
    }
    return url.toString();
  }

  private async connect(gen: number) {
    if (this.stopped || gen !== this.generation || typeof window === "undefined") {
      return;
    }
    this.setStatus(this.attempt === 0 ? "connecting" : "reconnecting");

    let token: WsTokenResponse;
    try {
      token = await this.fetchToken();
    } catch (e) {
      if (gen !== this.generation) return;
      if (e instanceof UnauthorizedError) {
        // No valid session: retrying cannot succeed until the user logs in.
        this.setStatus("unauthorized");
        return;
      }
      this.scheduleReconnect(gen);
      return;
    }
    if (this.stopped || gen !== this.generation) return;

    let ws: WebSocket;
    try {
      ws = new WebSocket(this.buildUrl(token.url), ["bearer", token.token]);
    } catch {
      this.scheduleReconnect(gen);
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      this.setStatus("open");
      if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = setInterval(
        () => this.send({ action: "ping" }),
        HEARTBEAT_MS,
      );
    };

    ws.onmessage = (ev) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(typeof ev.data === "string" ? ev.data : "");
      } catch {
        return;
      }
      if (!parsed || typeof parsed !== "object") return;
      const m = parsed as Record<string, unknown>;
      if (typeof m.channel === "string" && "data" in m) {
        const msg: WsMessage = {
          channel: m.channel,
          data: m.data,
          timestamp: typeof m.timestamp === "string" ? m.timestamp : undefined,
        };
        for (const h of this.handlers) h(msg);
        return;
      }
      if (typeof m.error === "string" && process.env.NODE_ENV !== "production") {
        console.warn("[ws] server error:", m.error);
      }
      // ack / pong: nothing to do.
    };

    ws.onclose = (ev) => {
      if (this.ws === ws) this.ws = null;
      if (this.heartbeatTimer) {
        clearInterval(this.heartbeatTimer);
        this.heartbeatTimer = null;
      }
      if (this.stopped || gen !== this.generation) return;
      // 1008 = token rejected (expired between fetch and dial, or user
      // disabled); 4001 = the server closed an open socket because the
      // session it was opened with expired. Either way the next attempt
      // fetches a fresh token; if the session itself is gone, fetchToken
      // reports "unauthorized" and we stop.
      if (ev.code === POLICY_VIOLATION && process.env.NODE_ENV !== "production") {
        console.warn("[ws] handshake rejected:", ev.reason || "policy violation");
      }
      this.scheduleReconnect(gen);
    };

    ws.onerror = () => {
      // onclose always follows; reconnect is handled there.
    };
  }

  private scheduleReconnect(gen: number) {
    if (this.stopped || gen !== this.generation || this.reconnectTimer) return;
    this.setStatus("reconnecting");
    // Exponential backoff with full jitter.
    const cap = Math.min(MAX_BACKOFF_MS, MIN_BACKOFF_MS * 2 ** this.attempt);
    const delay = Math.max(MIN_BACKOFF_MS / 2, Math.random() * cap);
    this.attempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.connect(gen);
    }, delay);
  }
}

let singleton: WsClient | null = null;

export function getWsClient(): WsClient {
  if (!singleton) singleton = new WsClient();
  return singleton;
}
