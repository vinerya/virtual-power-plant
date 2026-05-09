import { NextResponse } from "next/server";
import { AUTH_COOKIE_NAME } from "@/lib/api/client";

export async function POST() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set(AUTH_COOKIE_NAME, "", {
    httpOnly: true,
    path: "/",
    maxAge: 0,
  });
  return res;
}
