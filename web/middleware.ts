import { NextResponse, type NextRequest } from "next/server";
import {
  contentSecurityPolicy,
  generateNonce,
  hstsHeader,
} from "@/lib/security-headers";

const COOKIE = process.env.AUTH_COOKIE_NAME || "vpp_session";
const PUBLIC_PATHS = [
  "/login",
  "/api/auth/login",
  "/api/auth/logout",
  "/api/auth/me",
  // These return their own 401 JSON; must not be redirected to the HTML
  // login page.
  "/api/auth/ws-token",
  "/api/auth/password",
  "/api/auth/logout-all",
];

// Operator routes are everything else under /(operator); customer routes
// live under /portal. We rely on a token "audience" claim to route users:
//   - aud == "customer"  → /portal
//   - otherwise          → operator console
//
// The decode here is *advisory only* (best-effort base64 of the JWT
// payload) — the real check is the per-route /api/auth/me lookup which
// asks FastAPI to validate signature + audience server-side.
function decodeAudience(token: string): string | null {
  const parts = token.split(".");
  if (parts.length < 2) return null;
  try {
    const payload = JSON.parse(
      Buffer.from(parts[1].replace(/-/g, "+").replace(/_/g, "/"), "base64").toString(
        "utf-8",
      ),
    );
    if (typeof payload.aud === "string") return payload.aud;
    return null;
  } catch {
    return null;
  }
}

function route(request: NextRequest, next: () => NextResponse): NextResponse {
  const { pathname } = request.nextUrl;

  if (
    PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/")) ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon") ||
    pathname.startsWith("/static") ||
    // Self-hosted Monaco editor assets (public/monaco, not sensitive).
    pathname.startsWith("/monaco/")
  ) {
    return next();
  }

  const token = request.cookies.get(COOKIE)?.value;
  if (!token) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("from", pathname);
    return NextResponse.redirect(url);
  }

  const aud = decodeAudience(token);
  const isCustomerRoute =
    pathname === "/portal" || pathname.startsWith("/portal/");

  if (aud === "customer" && !isCustomerRoute) {
    const url = request.nextUrl.clone();
    url.pathname = "/portal";
    url.search = "";
    return NextResponse.redirect(url);
  }

  return next();
}

export function middleware(request: NextRequest) {
  // Per-request CSP nonce: Next.js reads it from the *request* header and
  // stamps it on the scripts it renders; the browser enforces the response
  // header. See lib/security-headers.ts.
  const nonce = generateNonce();
  const csp = contentSecurityPolicy(nonce);

  const response = route(request, () => {
    const headers = new Headers(request.headers);
    headers.set("content-security-policy", csp);
    headers.set("x-nonce", nonce);
    return NextResponse.next({ request: { headers } });
  });
  response.headers.set("Content-Security-Policy", csp);
  const hsts = hstsHeader();
  if (hsts) response.headers.set("Strict-Transport-Security", hsts);
  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
