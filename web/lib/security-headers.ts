// Security headers for the console. Imported by middleware.ts (edge
// runtime) and by route handlers, so no Node-only APIs here.
//
// The Content-Security-Policy is built per request with a fresh nonce:
// Next.js reads the nonce from the request's `content-security-policy`
// header (set by middleware.ts) and puts it on the inline scripts it
// renders (the RSC payload, the bootstrap). 'strict-dynamic' then trusts the
// chunks those scripts load (webpack chunks, Monaco's AMD loader under
// /monaco/vs). Pages are rendered per request (`dynamic = "force-dynamic"`
// in app/layout.tsx) so every response carries the nonce of its CSP.
//
// The static headers (nosniff, framing, referrer, permissions) are set in
// next.config.ts `headers()`, which also covers the /_next/static assets the
// middleware skips.

const BACKEND =
  process.env.API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";

/** MapLibre style, tiles, sprites and glyphs (components/sites/site-map.tsx). */
export const MAP_TILE_ORIGIN = "https://tiles.openfreemap.org";

/**
 * Public WebSocket URL the *browser* dials (it cannot go through the
 * Next.js server). Resolution order:
 *   1. WS_PUBLIC_URL        (runtime, server-side env)
 *   2. NEXT_PUBLIC_WS_URL   (build-time)
 *   3. API_BASE_URL with http(s) → ws(s) + /api/v1/ws
 */
export function publicWsUrl(): string {
  const explicit = process.env.WS_PUBLIC_URL || process.env.NEXT_PUBLIC_WS_URL;
  if (explicit) return explicit;
  const u = new URL(BACKEND);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = `${u.pathname.replace(/\/$/, "")}/api/v1/ws`;
  u.search = "";
  return u.toString();
}

/** `ws(s)://host[:port]` of the WebSocket URL, or null when it is relative
 * (same origin, already covered by 'self'). */
function wsOrigin(): string | null {
  try {
    const u = new URL(publicWsUrl());
    return `${u.protocol}//${u.host}`;
  } catch {
    return null;
  }
}

/** Space-separated extra sources from CSP_CONNECT_SRC (runtime env). */
function extraConnectSources(): string[] {
  return (process.env.CSP_CONNECT_SRC || "").split(/[\s,]+/).filter(Boolean);
}

export function generateNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

export function contentSecurityPolicy(nonce: string): string {
  const dev = process.env.NODE_ENV === "development";
  const connect = ["'self'", wsOrigin(), MAP_TILE_ORIGIN, ...extraConnectSources()].filter(
    (s): s is string => Boolean(s),
  );
  const directives: Record<string, string[]> = {
    "default-src": ["'self'"],
    // 'unsafe-eval' only for `next dev` (React Refresh / eval source maps).
    "script-src": [
      "'self'",
      `'nonce-${nonce}'`,
      "'strict-dynamic'",
      ...(dev ? ["'unsafe-eval'"] : []),
    ],
    // React `style={…}` attributes, Monaco, Recharts and sonner inject styles
    // at runtime; style injection cannot run script.
    "style-src": ["'self'", "'unsafe-inline'"],
    "img-src": ["'self'", "data:", "blob:", MAP_TILE_ORIGIN],
    "font-src": ["'self'", "data:"],
    "connect-src": Array.from(new Set(connect)),
    // MapLibre and Monaco run web workers from blob: URLs.
    "worker-src": ["'self'", "blob:"],
    "child-src": ["'self'", "blob:"],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'none'"],
  };
  return Object.entries(directives)
    .map(([k, v]) => `${k} ${v.join(" ")}`)
    .join("; ");
}

/** HSTS for the console when HSTS_ENABLED=1 (runtime env; off by default
 * because TLS is usually terminated by a reverse proxy, which should set it). */
export function hstsHeader(): string | null {
  const on = process.env.HSTS_ENABLED;
  if (on !== "1" && on !== "true") return null;
  const maxAge = Number(process.env.HSTS_MAX_AGE_SECONDS || 31536000);
  return `max-age=${Number.isFinite(maxAge) ? Math.max(0, Math.floor(maxAge)) : 31536000}`;
}
