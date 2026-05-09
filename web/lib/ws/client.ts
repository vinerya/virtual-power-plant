// Singleton WebSocket client with auto-reconnect.
// Multiplexes channels: resource_updates, optimization_events, alerts.

export interface WsMessage {
  channel: string;
  data: unknown;
  timestamp?: string;
}

type Handler = (msg: WsMessage) => void;

interface StartOptions {
  channels: string[];
}

class WsClient {
  private ws: WebSocket | null = null;
  private handlers = new Set<Handler>();
  private channels: string[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private backoffMs = 1000;
  private stopped = true;

  start(opts: StartOptions) {
    this.channels = opts.channels;
    this.stopped = false;
    this.connect();
  }

  stop() {
    this.stopped = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* noop */
      }
      this.ws = null;
    }
  }

  on(handler: Handler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  send(msg: unknown) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
  }

  private url(): string {
    if (typeof window === "undefined") return "";
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const ch = encodeURIComponent(this.channels.join(","));
    return `${proto}//${host}/api/proxy/api/v1/ws?channels=${ch}`;
  }

  private connect() {
    if (this.stopped || typeof window === "undefined") return;
    try {
      this.ws = new WebSocket(this.url());
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.backoffMs = 1000;
    };
    this.ws.onmessage = (ev) => {
      let parsed: WsMessage | null = null;
      try {
        parsed = JSON.parse(ev.data) as WsMessage;
      } catch {
        return;
      }
      if (!parsed) return;
      for (const h of this.handlers) h(parsed);
    };
    this.ws.onclose = () => {
      this.ws = null;
      this.scheduleReconnect();
    };
    this.ws.onerror = () => {
      try {
        this.ws?.close();
      } catch {
        /* noop */
      }
    };
  }

  private scheduleReconnect() {
    if (this.stopped) return;
    if (this.reconnectTimer) return;
    const delay = Math.min(this.backoffMs, 30_000);
    this.backoffMs = Math.min(this.backoffMs * 2, 30_000);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }
}

let singleton: WsClient | null = null;

export function getWsClient(): WsClient {
  if (!singleton) singleton = new WsClient();
  return singleton;
}
