import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { z } from "zod";
import { AUTH_COOKIE_NAME, serverFetch } from "@/lib/api/client";
import type { Token } from "@/lib/api/types";

const bodySchema = z.object({
  current_password: z.string().min(1),
  new_password: z.string().min(1),
});

/**
 * Change the signed-in user's password. The backend revokes every session
 * of the account (this one included) and returns a fresh token, which
 * replaces the httpOnly session cookie here so the user stays signed in.
 */
export async function POST(request: Request) {
  const parsed = bodySchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ detail: "Invalid request body" }, { status: 400 });
  }
  const token = (await cookies()).get(AUTH_COOKIE_NAME)?.value;
  if (!token) return NextResponse.json({ detail: "Not signed in" }, { status: 401 });

  try {
    const fresh = await serverFetch<Token>("/api/v1/auth/password", {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
      body: JSON.stringify(parsed.data),
    });
    const res = NextResponse.json({ ok: true });
    res.cookies.set(AUTH_COOKIE_NAME, fresh.access_token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: fresh.expires_in,
    });
    return res;
  } catch (err) {
    const e = err as { status?: number; detail?: unknown };
    const status = typeof e.status === "number" ? e.status : 502;
    const detail =
      e.detail && typeof e.detail === "object" && "detail" in e.detail
        ? (e.detail as { detail: unknown }).detail
        : "Password change failed";
    return NextResponse.json({ detail }, { status });
  }
}
