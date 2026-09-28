import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";
const NOW = Date.now();

const RUNS = [
  {
    id: "run-1",
    problem_type: "stochastic",
    status: "optimal",
    objective_value: 1234.5,
    solve_time_ms: 42,
    fallback_used: false,
    created_at: new Date(NOW - 60_000).toISOString(),
    inputs: { prices: [0.1, 0.2] },
    solution: {
      charge: [10, 0, 0, 0],
      discharge: [0, 0, 5, 12],
      power: [10, 0, -5, -12],
    },
    metadata: { rationale: "Charged off-peak; discharged during evening peak." },
  },
];

const EXPLAIN = {
  actual: {
    name: "actual",
    total_cost: 110.5,
    per_step: [
      { step: 0, charge: 10, discharge: 0, cost: 1 },
      { step: 1, charge: 0, discharge: 0, cost: 0 },
      { step: 2, charge: 0, discharge: 5, cost: -2 },
      { step: 3, charge: 0, discharge: 12, cost: -5 },
    ],
  },
  counterfactuals: [
    {
      name: "no_action",
      total_cost: 145.0,
      per_step: [
        { step: 0, charge: 0, discharge: 0 },
        { step: 1, charge: 0, discharge: 0 },
        { step: 2, charge: 0, discharge: 0 },
        { step: 3, charge: 0, discharge: 0 },
      ],
    },
    {
      name: "price_naive",
      total_cost: 132.0,
      per_step: [
        { step: 0, charge: 5, discharge: 0 },
        { step: 1, charge: 5, discharge: 0 },
        { step: 2, charge: 0, discharge: 5 },
        { step: 3, charge: 0, discharge: 5 },
      ],
    },
  ],
  binding_constraints: [
    { step: 2, name: "soc_upper", slack: 0.0, description: "Battery hit upper SOC at t=2" },
    { step: 3, name: "demand_ceiling", slack: 0.001, description: "Demand-charge ceiling reached" },
  ],
  rationale: "Pre-charge at t=0 to capture peak arbitrage at t=2,3.",
};

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
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) }),
  );
  await context.route(/\/api\/proxy\/api\/v1\/resources\/?(\?.*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify([]) }),
  );
  await context.route("**/api/proxy/api/v1/optimization/history**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(RUNS) }),
  );
  await context.route("**/api/proxy/api/v1/dispatches/run-1/explain", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(EXPLAIN) }),
  );
});

test("opens dispatch sheet, navigates to Counterfactual, sees comparison chart and binding constraints", async ({
  page,
}) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("pw");
  await page.getByRole("button", { name: /sign in/i }).click();

  await page.goto("/trading/dispatches");
  await page.getByTestId("dispatch-rows").locator("tr").first().click();

  await page.getByRole("tab", { name: /counterfactual/i }).click();
  await expect(page.getByTestId("explainer-content")).toBeVisible();
  await expect(page.getByTestId("cost-comparison-chart")).toBeVisible();
  await expect(page.getByTestId("binding-constraints")).toContainText(
    "upper SOC",
  );
  await expect(page.getByTestId("explainer-content")).toContainText(
    /pre-charge/i,
  );
});

test("shows empty state when explainer 404s", async ({ page, context }) => {
  await context.route(
    "**/api/proxy/api/v1/dispatches/run-1/explain",
    (route) =>
      route.fulfill({
        status: 404,
        contentType: "application/json",
        body: JSON.stringify({ detail: "not found" }),
      }),
  );

  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("pw");
  await page.getByRole("button", { name: /sign in/i }).click();

  await page.goto("/trading/dispatches");
  await page.getByTestId("dispatch-rows").locator("tr").first().click();
  await page.getByRole("tab", { name: /counterfactual/i }).click();
  await expect(page.getByTestId("explainer-empty")).toBeVisible();
});
