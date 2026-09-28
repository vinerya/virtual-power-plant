import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, serverFetch } from "@/lib/api/client";
import { forwardedClientHeaders } from "@/lib/api/forwarded";

interface MeResponse {
  username?: string;
  role?: string;
  audience?: "operator" | "customer" | string;
  scopes?: string[];
}

/**
 * Verifies the audience claim of the current session by asking FastAPI to
 * validate the JWT (signature + expiry + audience). The middleware does an
 * advisory base64 decode for routing; this endpoint is the authoritative
 * check whenever the UI needs to gate per-page behavior.
 */
export async function GET(request: Request) {
  const cookieStore = await cookies();
  const token = cookieStore.get(AUTH_COOKIE_NAME)?.value;
  if (!token) return NextResponse.json({ authenticated: false }, { status: 401 });

  try {
    const me = await serverFetch<MeResponse>("/api/v1/auth/me", {
      headers: {
        ...forwardedClientHeaders(request.headers),
        authorization: `Bearer ${token}`,
      },
    });
    return NextResponse.json({
      authenticated: true,
      audience: me.audience ?? "operator",
      username: me.username,
      // Used for role-aware UI (e.g. hiding the order ticket from viewers);
      // the backend still enforces every permission.
      role: me.role,
      scopes: me.scopes,
    });
  } catch (err) {
    const status =
      typeof err === "object" && err && "status" in err
        ? (err as { status: number }).status
        : 500;
    return NextResponse.json(
      { authenticated: false },
      { status: status === 401 ? 401 : 500 },
    );
  }
}
