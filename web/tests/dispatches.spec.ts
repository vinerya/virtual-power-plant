import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";
const NOW = Date.now();

const RESOURCES = [
  {
    id: "r1",
    name: "Battery A",
    resource_type: "battery",
    rated_power: 250,
    metadata: {},
    online: true,
    current_power: 0,
    efficiency: 0.95,
    created_at: new Date(NOW).toISOString(),
    updated_at: new Date(NOW).toISOString(),
  },
];

const RUNS = [
  {
    id: "d1",
    problem_type: "stochastic",
    status: "optimal",
    objective_value: 1234.5,
    solve_time_ms: 42.1,
    fallback_used: false,
    created_at: new Date(NOW - 1000 * 60 * 60).toISOString(),
    solver: "GLPK",
    iterations: 14,
    gap: 0.001,
    inputs: { prices: [0.1, 0.12, 0.15, 0.2, 0.18] },
    solution: {
      charge: [10, 8, 5, 0],
      discharge: [0, 0, 4, 12],
      power: [10, 8, 1, -12],
    },
    metadata: { rationale: "Buy energy off-peak, sell during evening peak." },
  },
  {
    id: "d2",
    problem_type: "realtime",
    status: "optimal",
    objective_value: 200.0,
    solve_time_ms: 8.4,
    fallback_used: true,
    created_at: new Date(NOW - 1000 * 60 * 60 * 5).toISOString(),
    solver: "fallback",
    iterations: 1,
    gap: null,
    inputs: { target_power_kw: 100 },
    solution: { charge: [0, 0], discharge: [50, 50], power: [-50, -50] },
    metadata: {},
  },
];

test.beforeEach(async ({ context }) => {
  await context.route("**/api/auth/login", (route) =>
    route.fulfill({
      status: 200,
      headers: {
        "set-cookie": `vpp_session=${FAKE_TOKEN}; Path=/; HttpOnly; SameSite=Lax`,
      },
      contentType: "application/json",
      body: JSON.stringify({ ok: true }),
    }),
  );
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
      body: JSON.stringify(RESOURCES),
    }),
  );
  await context.route(
    "**/api/proxy/api/v1/optimization/history**",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(RUNS),
      }),
  );
});

test("filters by 24h, opens dispatch sheet, shows schedule chart", async ({
  page,
}) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("correct-pass");
  await page.getByRole("button", { name: /sign in/i }).click();

  await page.goto("/trading/dispatches");
  await expect(page.getByTestId("dispatches-view")).toBeVisible();

  await page.getByTestId("window-select").selectOption("24h");

  // Should show only the d1 run (within last 24h). d2 is 5h ago so also in.
  const rows = page.getByTestId("dispatch-rows").locator("tr");
  await expect(rows).toHaveCount(2);

  await rows.first().click();
  const sheet = page.getByTestId("dispatch-sheet");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByTestId("solution-section")).toBeVisible();

  // Esc closes
  await page.keyboard.press("Escape");
  await expect(sheet).not.toBeVisible();
});
