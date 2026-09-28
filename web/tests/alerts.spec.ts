import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";

const NOW = Date.now();
const ALERTS = [
  {
    id: "a1",
    timestamp: new Date(NOW - 60_000 * 5).toISOString(),
    severity: "critical",
    source: "res-1",
    source_kind: "resource",
    source_link: "/assets/res-1",
    title: "Battery offline",
    message: "Heartbeat missing",
    status: "active",
  },
  {
    id: "a2",
    timestamp: new Date(NOW - 60_000 * 30).toISOString(),
    severity: "warning",
    source: "res-2",
    source_kind: "resource",
    source_link: "/assets/res-2",
    title: "SOH degraded",
    message: "Below 0.85",
    status: "active",
  },
  {
    id: "a3",
    timestamp: new Date(NOW - 60_000 * 90).toISOString(),
    severity: "info",
    source: "system",
    title: "Tariff updated",
    message: "Schedule edited",
    status: "active",
  },
];

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
    r.fulfill({ status: 200, body: JSON.stringify({ status: "ok" }) }),
  );

  await context.route(/\/api\/proxy\/api\/v1\/alerts(\?.*)?$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(ALERTS),
    });
  });

  await context.route(
    /\/api\/proxy\/api\/v1\/alerts\/[^/]+\/ack$/,
    async (route) => {
      const url = route.request().url();
      const id = url.split("/").slice(-2, -1)[0];
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ...ALERTS.find((a) => a.id === id),
          status: "acknowledged",
        }),
      });
    },
  );

  await context.route(
    /\/api\/proxy\/api\/v1\/alerts\/[^/]+\/snooze$/,
    async (route) => {
      const url = route.request().url();
      const id = url.split("/").slice(-2, -1)[0];
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ...ALERTS.find((a) => a.id === id),
          status: "snoozed",
        }),
      });
    },
  );
});

test("filters by critical, ack one, snooze another, sparkline updates", async ({
  page,
}) => {
  await page.goto("/alerts");

  await expect(page.getByTestId("severity-sparkline")).toBeVisible();
  await expect(page.getByTestId("alert-rows").getByTestId("alert-row")).toHaveCount(
    3,
  );

  // Filter to critical only.
  await page.getByTestId("filter-critical").click();
  const rows = page.getByTestId("alert-rows").getByTestId("alert-row");
  await expect(rows).toHaveCount(1);
  await expect(rows.first()).toHaveAttribute("data-severity", "critical");

  // Ack the critical alert.
  await rows.first().getByTestId("ack-button").click();
  await expect(rows.first()).toHaveAttribute("data-status", /acknowledged|active/);

  // Switch to warning, snooze for 1h.
  await page.getByTestId("filter-warning").click();
  const warnRow = page.getByTestId("alert-rows").getByTestId("alert-row").first();
  await warnRow.getByTestId("snooze-picker").locator("button").first().click();
  await page.getByTestId("snooze-1h").click();
  await expect(warnRow).toHaveAttribute("data-status", /snoozed|active/);
});
