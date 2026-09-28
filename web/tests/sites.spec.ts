import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";

const SITES = [
  {
    id: "site-a",
    name: "Site Alpha",
    lat: 37.77,
    lon: -122.42,
    resource_ids: ["r1", "r2"],
    total_resources: 2,
    online_count: 2,
    current_power: 350,
    rated_power: 600,
    active_alerts: 0,
    health: "green",
  },
  {
    id: "site-b",
    name: "Site Bravo",
    lat: 40.71,
    lon: -74.0,
    resource_ids: ["r3"],
    total_resources: 1,
    online_count: 0,
    current_power: 0,
    rated_power: 200,
    active_alerts: 3,
    health: "red",
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
  await context.route(/\/api\/proxy\/api\/v1\/sites$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(SITES),
    }),
  );
  // Block external tile requests.
  await context.route(/tiles\.openfreemap\.org/, (r) =>
    r.fulfill({ status: 200, body: "{}" }),
  );
});

test("renders sidebar list, click row selects site (synced with map)", async ({
  page,
}) => {
  await page.goto("/sites");

  // Sidebar list rendered.
  const rows = page.getByTestId("site-list").getByTestId("site-row");
  await expect(rows).toHaveCount(2);

  // Click a site row, it becomes selected.
  await rows.first().click();
  await expect(rows.first()).toHaveAttribute("aria-selected", "true");

  // KPI strip is rendered.
  await expect(page.getByText("Total sites")).toBeVisible();
});
