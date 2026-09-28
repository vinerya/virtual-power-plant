import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, serverFetch } from "@/lib/api/client";

/**
 * "Log out everywhere": revoke every session token of the signed-in user on
 * the backend, then clear this browser's session cookie.
 */
export async function POST() {
  const token = (await cookies()).get(AUTH_COOKIE_NAME)?.value;
  if (!token) return NextResponse.json({ detail: "Not signed in" }, { status: 401 });
  try {
    await serverFetch("/api/v1/auth/logout-all", {
      method: "POST",
      headers: { authorization: `Bearer ${token}` },
    });
  } catch (err) {
    const status = (err as { status?: number }).status;
    // 401: the session was already revoked -- clearing the cookie is right.
    if (status !== 401) {
      return NextResponse.json(
        { detail: "Could not revoke sessions" },
        { status: typeof status === "number" ? status : 502 },
      );
    }
  }
  const res = NextResponse.json({ ok: true });
  res.cookies.set(AUTH_COOKIE_NAME, "", { httpOnly: true, path: "/", maxAge: 0 });
  return res;
}
