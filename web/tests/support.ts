import type { BrowserContext, Route } from "@playwright/test";

/**
 * Shared setup for specs that stub the backend: a session cookie, the
 * `/api/auth/me` role lookup used for role-aware UI, and a healthy backend.
 * The WebSocket token route answers 401 unless a spec overrides it, so the
 * live client parks in "signed out" instead of retrying.
 */
export async function signInAs(
  context: BrowserContext,
  role: "admin" | "operator" | "viewer" | "customer",
) {
  await context.addCookies([
    {
      name: "vpp_session",
      value: "fake.jwt.token",
      domain: "localhost",
      path: "/",
      httpOnly: true,
      sameSite: "Lax",
    },
  ]);
  await context.route("**/api/auth/me", (r) =>
    json(r, {
      authenticated: true,
      audience: role === "customer" ? "customer" : "operator",
      username: `${role}-user`,
      role,
    }),
  );
  await context.route("**/api/proxy/health", (r) => json(r, { status: "ok" }));
  await context.route("**/api/auth/ws-token", (r) => json(r, { detail: "no" }, 401));
}

export function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

/** Regex matching a proxied backend path, with or without a query string. */
export function api(path: string): RegExp {
  const esc = path.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`/api/proxy${esc}/?(\\?.*)?$`);
}
