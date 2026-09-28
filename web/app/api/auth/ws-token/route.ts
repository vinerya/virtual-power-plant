import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, serverFetch } from "@/lib/api/client";

// Always evaluate per request (reads cookies + env at runtime).
export const dynamic = "force-dynamic";

interface BackendWsToken {
  token: string;
  expires_in: number;
  channels?: string[];
}

const BACKEND =
  process.env.API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";

/**
 * Public WebSocket URL the *browser* should dial. Next.js route handlers
 * cannot proxy a WebSocket upgrade, so the browser connects to FastAPI
 * directly. Resolution order:
 *   1. WS_PUBLIC_URL        (runtime, server-side env)
 *   2. NEXT_PUBLIC_WS_URL   (build-time)
 *   3. API_BASE_URL with http(s) → ws(s) + /api/v1/ws
 * Set (1) or (2) whenever API_BASE_URL is an internal address the browser
 * cannot reach (e.g. `http://api:8000` inside docker-compose).
 */
function publicWsUrl(): string {
  const explicit = process.env.WS_PUBLIC_URL || process.env.NEXT_PUBLIC_WS_URL;
  if (explicit) return explicit;
  const u = new URL(BACKEND);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = `${u.pathname.replace(/\/$/, "")}/api/v1/ws`;
  u.search = "";
  return u.toString();
}

/**
 * Exchange the httpOnly session cookie for a short-lived, socket-only token
 * (FastAPI `POST /api/v1/ws/token`). The long-lived session JWT never
 * reaches client-side JavaScript.
 */
export async function GET() {
  const cookieStore = await cookies();
  const session = cookieStore.get(AUTH_COOKIE_NAME)?.value;
  if (!session) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  try {
    const t = await serverFetch<BackendWsToken>("/api/v1/ws/token", {
      method: "POST",
      headers: { authorization: `Bearer ${session}` },
    });
    return NextResponse.json(
      { token: t.token, expires_in: t.expires_in, url: publicWsUrl() },
      { headers: { "cache-control": "no-store" } },
    );
  } catch (err) {
    const status =
      typeof err === "object" && err && "status" in err
        ? (err as { status: number }).status
        : 502;
    // 401 → session expired; anything else → backend problem.
    return NextResponse.json(
      { detail: status === 401 ? "Session expired" : "Could not obtain a WebSocket token" },
      { status: status === 401 ? 401 : 502 },
    );
  }
}
