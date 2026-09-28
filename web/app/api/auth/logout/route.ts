import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, serverFetch } from "@/lib/api/client";
import { forwardedClientHeaders } from "@/lib/api/forwarded";

/**
 * Sign out of this browser: tell the API (so the sign-out lands in the audit
 * log) and clear the session cookie. The backend call is best effort -- an
 * expired or revoked token must not keep anyone signed in.
 */
export async function POST(request: NextRequest) {
  const token = (await cookies()).get(AUTH_COOKIE_NAME)?.value;
  if (token) {
    try {
      await serverFetch("/api/v1/auth/logout", {
        method: "POST",
        headers: {
          ...forwardedClientHeaders(request.headers),
          authorization: `Bearer ${token}`,
        },
        signal: AbortSignal.timeout(5000),
      });
    } catch {
      // Already invalid, or the API is down: clearing the cookie is still right.
    }
  }
  const res = NextResponse.json({ ok: true });
  res.cookies.set(AUTH_COOKIE_NAME, "", {
    httpOnly: true,
    path: "/",
    maxAge: 0,
  });
  return res;
}
