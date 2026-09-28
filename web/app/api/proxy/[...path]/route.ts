import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { AUTH_COOKIE_NAME } from "@/lib/api/client";

const BACKEND =
  process.env.API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const url = new URL(request.url);
  const target = `${BACKEND}/${path.join("/")}${url.search}`;

  const cookieStore = await cookies();
  const token = cookieStore.get(AUTH_COOKIE_NAME)?.value;

  const headers = new Headers();
  const incomingCT = request.headers.get("content-type");
  if (incomingCT) headers.set("content-type", incomingCT);
  if (token) headers.set("authorization", `Bearer ${token}`);

  const init: RequestInit = {
    method: request.method,
    headers,
    cache: "no-store",
  };

  if (!["GET", "HEAD"].includes(request.method)) {
    init.body = await request.text();
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch {
    // Backend down / DNS failure / connection refused. Return a JSON 502
    // so the UI can render a meaningful error instead of a Next.js 500 page.
    return NextResponse.json(
      { detail: "API server unreachable" },
      { status: 502 },
    );
  }
  const body = await upstream.text();
  const res = new NextResponse(body, {
    status: upstream.status,
    headers: {
      "content-type":
        upstream.headers.get("content-type") || "application/json",
    },
  });
  return res;
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const DELETE = proxy;
export const PATCH = proxy;
