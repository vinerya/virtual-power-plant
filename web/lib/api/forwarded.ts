// Client-address headers for server-side calls from Route Handlers to FastAPI.
//
// Without them every console request reaches the API from this server's
// address, and the API's per-IP rate limiter sees a single client. Next.js
// fills `x-forwarded-for` with the socket peer when the request arrived
// without one; a reverse proxy in front of the console appends to it.
//
// The API honours these headers only from addresses in VPP_TRUSTED_PROXIES
// and walks X-Forwarded-For from the right, so a client-supplied value can
// add entries but cannot replace the hop this server (or its proxy) saw.
// If the console is reachable directly (no reverse proxy that sets
// X-Forwarded-For), a client can still choose the address it is rate-limited
// under: put a proxy in front, or leave VPP_TRUSTED_PROXIES empty.

export function forwardedClientHeaders(incoming: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  const xff = incoming
    .get("x-forwarded-for")
    ?.split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (xff && xff.length > 0) {
    out["x-forwarded-for"] = xff.join(", ");
    // The nearest hop: what this server (or the proxy in front of it) saw.
    out["x-real-ip"] = xff[xff.length - 1];
  }
  return out;
}
