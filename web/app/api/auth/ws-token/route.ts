import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, serverFetch } from "@/lib/api/client";
import { forwardedClientHeaders } from "@/lib/api/forwarded";
import { publicWsUrl } from "@/lib/security-headers";

// Always evaluate per request (reads cookies + env at runtime).
export const dynamic = "force-dynamic";

interface BackendWsToken {
  token: string;
  expires_in: number;
  channels?: string[];
}

/**
 * Exchange the httpOnly session cookie for a short-lived, socket-only token
 * (FastAPI `POST /api/v1/ws/token`). The long-lived session JWT never
 * reaches client-side JavaScript.
 */
export async function GET(request: Request) {
  const cookieStore = await cookies();
  const session = cookieStore.get(AUTH_COOKIE_NAME)?.value;
  if (!session) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  try {
    const t = await serverFetch<BackendWsToken>("/api/v1/ws/token", {
      method: "POST",
      headers: {
        ...forwardedClientHeaders(request.headers),
        authorization: `Bearer ${session}`,
      },
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
