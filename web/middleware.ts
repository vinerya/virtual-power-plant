import { NextResponse, type NextRequest } from "next/server";

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

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (
    PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/")) ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon") ||
    pathname.startsWith("/static") ||
    // Self-hosted Monaco editor assets (public/monaco, not sensitive).
    pathname.startsWith("/monaco/")
  ) {
    return NextResponse.next();
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

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
