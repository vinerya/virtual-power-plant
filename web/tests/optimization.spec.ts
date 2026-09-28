import { test, expect } from "@playwright/test";
import { api, json, signInAs } from "./support";

const NOW = new Date().toISOString();

const RESOURCES = [
  {
    id: "bat-1",
    name: "Battery A",
    resource_type: "battery",
    rated_power: 250,
    metadata: {},
    online: true,
    current_power: 0,
    efficiency: 0.95,
    created_at: NOW,
    updated_at: NOW,
  },
  {
    id: "pv-1",
    name: "Roof PV",
    resource_type: "solar",
    rated_power: 100,
    metadata: {},
    online: true,
    current_power: 40,
    efficiency: 0.2,
    created_at: NOW,
    updated_at: NOW,
  },
];

const SCHEDULE = {
  run_id: "run-sched-1",
  status: "success",
  method: "milp_highs",
  fallback_used: false,
  fallback_reason: null,
  solve_time_ms: 49.5,
  objective_value: -102.7,
  energy_cost: -90.13,
  wear_cost: 35.91,
  interval_minutes: 60,
  prices: [0.1, 0.1, 0.3, 0.25],
  charge: [250, 200, 0, 0],
  discharge: [0, 0, 250, 200],
  power: [250, 200, -250, -200],
  per_resource: {
    "bat-1": { charge: [250, 200, 0, 0], discharge: [0, 0, 250, 200], soc: [0.7, 0.9, 0.6, 0.4] },
  },
  resources: [{ id: "bat-1", name: "Battery A", resource_type: "battery" }],
  tariff_id: null,
  terminal_soc_policy: "value",
  notes: [],
};

const BACKTEST = {
  run_id: "run-bt-1",
  status: "success",
  ticks: 4,
  realized_cost: -34.5,
  realized_cost_adjusted: -27.4,
  no_action_cost: 0,
  rules_cost: -24.1,
  rules_cost_adjusted: -24.5,
  perfect_foresight_cost_adjusted: -32.3,
  perfect_foresight_status: "success",
  regret: 4.92,
  terminal_energy_value_per_kwh: 0.1588,
  terminal_soc_policy: "value",
  final_soc: 0.05,
  fallback_count: 0,
  cumulative_solve_time_ms: 402.7,
  cumulative_solver_iterations: 174,
  wall_time_s: 0.4,
  forecast_mode: "persistence",
  interval_minutes: 60,
  horizon_steps: 24,
  charge: [47, 0, 0, 0],
  discharge: [0, 0, 50, 35],
  power: [47, 0, -50, -35],
  soc: [0.95, 0.95, 0.42, 0.05],
  notes: ["costs count only the battery's net energy exchange (idle = 0)"],
};

test.beforeEach(async ({ context }) => {
  await signInAs(context, "operator");
  await context.route(api("/api/v1/resources"), (r) => json(r, RESOURCES));
  await context.route(api("/api/v1/tariffs"), (r) =>
    json(r, [{ id: "tou-1", name: "E-TOU-C", utility: "PG&E" }]),
  );
});

test("plans a schedule from prices and links the run to the explainer", async ({ page }) => {
  let sent: Record<string, unknown> | null = null;
  await page.route(api("/api/v1/optimization/schedule"), (r) => {
    sent = r.request().postDataJSON();
    return json(r, SCHEDULE);
  });
  await page.route(api("/api/v1/dispatches/run-sched-1"), (r) =>
    json(r, {
      id: "run-sched-1",
      problem_type: "schedule",
      status: "success",
      created_at: NOW,
      inputs: { prices: SCHEDULE.prices },
      solution: { charge: SCHEDULE.charge, discharge: SCHEDULE.discharge, power: SCHEDULE.power },
      metadata: {},
    }),
  );
  await page.route(api("/api/v1/optimization/history"), (r) => json(r, []));
  await page.route(/\/api\/proxy\/api\/v1\/dispatches\/run-sched-1\/explain$/, (r) =>
    r.fulfill({ status: 204 }),
  );

  await page.goto("/optimization");
  // Only batteries are offered.
  const batteries = page.getByLabel("Batteries");
  await expect(batteries.locator("option")).toHaveCount(1);
  await batteries.selectOption("bat-1");
  await page.getByLabel(/Prices \(currency per kWh/).fill("0.1, 0.1\n0.3 0.25");
  await page.getByTestId("schedule-run").click();

  const result = page.getByTestId("schedule-result");
  await expect(result).toBeVisible();
  await expect(page.getByTestId("stat-energy-cost")).toContainText("-90.13");
  await expect(page.getByTestId("power-price-chart")).toBeVisible();
  await expect(page.getByTestId("soc-chart")).toBeVisible();
  expect(sent).toMatchObject({
    prices: [0.1, 0.1, 0.3, 0.25],
    resource_ids: ["bat-1"],
    interval_minutes: 60,
    degradation_aware: true,
    terminal_soc_policy: "value",
  });

  await page.getByTestId("explain-link").click();
  await expect(page).toHaveURL(/\/trading\/dispatches\?run=run-sched-1&view=explain/);
  const sheet = page.getByTestId("dispatch-sheet");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByRole("tab", { name: "Counterfactual" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

test("uses a stored tariff and surfaces server validation errors", async ({ page }) => {
  let sent: Record<string, unknown> | null = null;
  await page.route(api("/api/v1/optimization/schedule"), (r) => {
    sent = r.request().postDataJSON();
    return json(r, { detail: "no online battery resources to schedule" }, 422);
  });
  await page.goto("/optimization");
  await page.getByTestId("source-tariff").check();
  await page.getByLabel("Tariff", { exact: true }).selectOption("tou-1");
  await page.getByLabel("Horizon (hours)").fill("48");
  await page.getByTestId("schedule-run").click();
  await expect(page.getByTestId("error-state")).toContainText(
    "no online battery resources to schedule",
  );
  expect(sent).toMatchObject({ tariff_id: "tou-1", horizon_hours: 48, nem: "nem2" });
  expect(sent).not.toHaveProperty("prices");
});

test("runs a closed-loop backtest and compares against baselines", async ({ page }) => {
  let sent: Record<string, unknown> | null = null;
  await page.route(api("/api/v1/optimization/backtest"), (r) => {
    sent = r.request().postDataJSON();
    return json(r, BACKTEST);
  });
  await page.goto("/optimization");
  await page.getByRole("tab", { name: /Backtest vs/ }).click();
  await page.getByTestId("bt-capacity").fill("200");
  await page.getByTestId("bt-forecast").selectOption("noisy");
  await page.getByTestId("bt-submit").click();

  await expect(page.getByTestId("backtest-result")).toBeVisible();
  await expect(page.getByTestId("stat-regret")).toContainText("4.92");
  const table = page.getByTestId("cost-table");
  await expect(table).toContainText("Rule-based");
  await expect(table).toContainText("-24.50");
  await expect(table).toContainText("Perfect foresight");
  await expect(page.getByTestId("explain-link")).toHaveAttribute(
    "href",
    "/trading/dispatches?run=run-bt-1&view=explain",
  );
  expect(sent).toMatchObject({
    battery: { capacity_kwh: 200, max_power_kw: 50, soc_init: 0.5 },
    forecast_mode: "noisy",
    noise_sigma: 0.1,
    seed: 42,
    horizon_steps: 24,
    compare_offline: true,
  });
  expect((sent as unknown as { prices: number[] }).prices).toHaveLength(48);
});
