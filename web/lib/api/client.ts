// Browser-side API client. All requests go through the Next.js
// `/api/proxy/*` route which attaches the httpOnly auth cookie.

export const AUTH_COOKIE_NAME =
  process.env.AUTH_COOKIE_NAME || "vpp_session";

const API_PREFIX = "/api/proxy";

export interface ApiError extends Error {
  status: number;
  detail?: unknown;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_PREFIX}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    let detail: unknown = text;
    try {
      detail = JSON.parse(text);
    } catch {
      // not json, keep raw text
    }
    const err = new Error(`API ${res.status} ${path}`) as ApiError;
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  if (res.status === 204) return undefined as unknown as T;
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("json")) return (await res.text()) as unknown as T;
  return (await res.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      body: body !== undefined ? JSON.stringify(body) : undefined,
    }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "PUT",
      body: body !== undefined ? JSON.stringify(body) : undefined,
    }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "PATCH",
      body: body !== undefined ? JSON.stringify(body) : undefined,
    }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

// Server-side fetch used by Route Handlers to talk directly to FastAPI.
const BACKEND =
  process.env.API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";

export async function serverFetch<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url = path.startsWith("http") ? path : `${BACKEND}${path}`;
  const res = await fetch(url, { ...init, cache: "no-store" });
  if (!res.ok) {
    const text = await res.text();
    let detail: unknown = text;
    try {
      detail = JSON.parse(text);
    } catch {
      // not json
    }
    const err = new Error(`API ${res.status} ${path}`) as ApiError;
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  if (res.status === 204) return undefined as unknown as T;
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("json")) return (await res.text()) as unknown as T;
  return (await res.json()) as T;
}
