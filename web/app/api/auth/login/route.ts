import { NextResponse } from "next/server";
import { z } from "zod";
import { AUTH_COOKIE_NAME, serverFetch } from "@/lib/api/client";
import { forwardedClientHeaders } from "@/lib/api/forwarded";
import type { Token } from "@/lib/api/types";

const bodySchema = z.object({
  username: z.string().min(1),
  password: z.string().min(1),
});

export async function POST(request: Request) {
  const json = await request.json().catch(() => null);
  const parsed = bodySchema.safeParse(json);
  if (!parsed.success) {
    return NextResponse.json(
      { detail: "Invalid request body" },
      { status: 400 },
    );
  }

  const { username, password } = parsed.data;

  // OAuth2 password grant as a form body -- never put credentials in the
  // query string (it ends up in proxy / server access logs).
  const form = new URLSearchParams({ grant_type: "password", username, password });

  try {
    const token = await serverFetch<Token>("/api/v1/auth/token", {
      method: "POST",
      headers: {
        ...forwardedClientHeaders(request.headers),
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: form.toString(),
    });
    const res = NextResponse.json({ ok: true });
    res.cookies.set(AUTH_COOKIE_NAME, token.access_token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: token.expires_in,
    });
    return res;
  } catch (err) {
    const status =
      typeof err === "object" && err && "status" in err
        ? (err as { status: number }).status
        : 500;
    return NextResponse.json(
      { detail: "Invalid credentials" },
      { status: status === 401 ? 401 : 500 },
    );
  }
}
