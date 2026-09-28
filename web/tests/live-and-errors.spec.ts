import { test, expect } from "@playwright/test";

/**
 * - Live updates: the WS client fetches a short-lived token from
 *   /api/auth/ws-token, dials the URL it returns with the initial channels,
 *   and turns backend broadcasts (EventBus envelope) into UI updates.
 * - No silent mocks: a failing backend endpoint renders an error state
 *   instead of demo data (mock mode is opt-in via NEXT_PUBLIC_USE_MOCKS).
 */

const FAKE_TOKEN = "fake.jwt.token";

test.beforeEach(async ({ context }) => {
  await context.addCookies([
    {
      name: "vpp_session",
      value: FAKE_TOKEN,
      domain: "localhost",
      path: "/",
      httpOnly: true,
      sameSite: "Lax",
    },
  ]);
  await context.route("**/api/proxy/health", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    }),
  );
  await context.route(/\/api\/proxy\/api\/v1\/resources\/?(\?.*)?$/, (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
});

test("connects to the WebSocket with a fetched token and shows broadcast alerts", async ({
  page,
}) => {
  let tokenRequests = 0;
  await page.route("**/api/auth/ws-token", (r) => {
    tokenRequests += 1;
    return r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        token: "short-lived-ws-token",
        expires_in: 60,
        url: "ws://backend.test/api/v1/ws",
      }),
    });
  });

  let resolveOpened: (v: { url: string; send: (m: string) => void }) => void;
  const opened = new Promise<{ url: string; send: (m: string) => void }>(
    (resolve) => {
      resolveOpened = resolve;
    },
  );
  await page.routeWebSocket(/backend\.test\/api\/v1\/ws/, (ws) => {
    ws.onMessage(() => {
      /* pings are ignored */
    });
    resolveOpened({ url: ws.url(), send: (m) => ws.send(m) });
  });

  await page.goto("/");
  const ws = await opened;
  expect(tokenRequests).toBeGreaterThanOrEqual(1);
  const url = new URL(ws.url);
  expect(url.searchParams.get("channels")?.split(",").sort()).toEqual([
    "alerts",
    "optimization_events",
    "resource_updates",
  ]);
  // The token travels in the subprotocol header, never in the URL.
  expect(url.searchParams.get("token")).toBeNull();

  await expect(page.getByTestId("live-status")).toHaveText("live");

  ws.send(
    JSON.stringify({
      channel: "alerts",
      timestamp: new Date().toISOString(),
      data: {
        event_id: "e1",
        event_type: "protocol_error",
        data: { message: "OpenADR VTN unreachable" },
        source: "openadr",
        severity: "error",
        timestamp: Date.now() / 1000,
      },
    }),
  );
  await expect(page.getByText("Protocol error")).toBeVisible();
  await expect(page.getByText("OpenADR VTN unreachable")).toBeVisible();
});

test("shows signed-out live status when the ws-token route returns 401", async ({
  page,
}) => {
  await page.route("**/api/auth/ws-token", (r) =>
    r.fulfill({
      status: 401,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Session expired" }),
    }),
  );
  await page.goto("/");
  await expect(page.getByTestId("live-status")).toHaveText("live: signed out");
});

test("alerts page shows an error state (not demo data) when the backend fails", async ({
  page,
}) => {
  await page.route(/\/api\/proxy\/api\/v1\/alerts(\?.*)?$/, (r) =>
    r.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Not Found" }),
    }),
  );
  await page.goto("/alerts");
  const err = page.getByTestId("error-state");
  await expect(err).toBeVisible({ timeout: 30_000 });
  await expect(err).toContainText("Failed to load alerts");
  await expect(err).toContainText("404");
  // None of the old built-in demo alerts leak through.
  await expect(page.getByText("Battery offline")).toHaveCount(0);
});

test("customer portal shows an error state when /customer/me is unavailable", async ({
  page,
}) => {
  await page.route(/\/api\/proxy\/api\/v1\/customer\/.*/, (r) =>
    r.fulfill({
      status: 502,
      contentType: "application/json",
      body: JSON.stringify({ detail: "API server unreachable" }),
    }),
  );
  await page.goto("/portal");
  const err = page.getByTestId("error-state");
  await expect(err).toBeVisible({ timeout: 30_000 });
  await expect(err).toContainText("unreachable");
  await expect(page.getByText("Demo Household")).toHaveCount(0);
});
