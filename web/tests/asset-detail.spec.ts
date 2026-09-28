import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";

const BATTERY = {
  id: "r1",
  name: "Battery A",
  resource_type: "battery",
  rated_power: 250,
  metadata: {},
  online: true,
  current_power: 120,
  efficiency: 0.95,
  capacity_kwh: 500,
  state_of_charge: 0.65,
  cycle_count: 412,
  charge_limit_kw: 200,
  discharge_limit_kw: 200,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

const SOLAR = {
  id: "r2",
  name: "Solar Field",
  resource_type: "solar",
  rated_power: 500,
  metadata: {},
  online: true,
  current_power: 320,
  efficiency: 0.92,
  irradiance: 850,
  dc_capacity_kw: 600,
  ac_capacity_kw: 500,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

test.beforeEach(async ({ context }) => {
  await context.route("**/api/auth/login", async (route) => {
    await route.fulfill({
      status: 200,
      headers: {
        "set-cookie": `vpp_session=${FAKE_TOKEN}; Path=/; HttpOnly; SameSite=Lax`,
      },
      contentType: "application/json",
      body: JSON.stringify({ ok: true }),
    });
  });

  await context.route("**/api/proxy/health", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    }),
  );

  await context.route(/\/api\/proxy\/api\/v1\/resources\/?(\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([BATTERY, SOLAR]),
    }),
  );

  await context.route("**/api/proxy/api/v1/resources/r1", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(BATTERY),
    }),
  );

  await context.route("**/api/proxy/api/v1/resources/r1/metrics**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        resource_id: "r1",
        window: "24h",
        points: Array.from({ length: 12 }).map((_, i) => ({
          timestamp: new Date(Date.now() - (12 - i) * 60_000).toISOString(),
          power: 100 + i * 10,
          state_of_charge: 0.5 + i * 0.02,
        })),
      }),
    }),
  );
});

test("navigates from fleet to asset detail and shows chart + battery panel", async ({
  page,
}) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("correct-pass");
  await page.getByRole("button", { name: /sign in/i }).click();

  await expect(page).toHaveURL("/");
  const link = page.getByRole("link", { name: "Battery A" });
  await expect(link).toBeVisible();
  await link.click();

  await expect(page).toHaveURL(/\/assets\/r1$/);
  await expect(page.getByTestId("asset-name")).toHaveText("Battery A");
  await expect(page.getByTestId("asset-detail")).toBeVisible();
  await expect(page.getByTestId("subtype-battery")).toBeVisible();
  await expect(page.getByText("State of charge — last 24h")).toBeVisible();
});
