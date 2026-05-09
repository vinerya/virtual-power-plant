import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";

const TARIFFS = [
  {
    id: "ci-tou",
    name: "C&I — TOU Demand",
    utility: "DemoCo",
    sector: "Commercial",
    source: "URDB",
  },
  {
    id: "resi-flat",
    name: "Residential Flat",
    utility: "DemoCo",
    sector: "Residential",
    source: "URDB",
  },
];

const TARIFF_CITOU = {
  ...TARIFFS[0],
  components: [
    { name: "Customer charge", kind: "fixed", unit: "$/mo", rate: 75 },
    { name: "Energy", kind: "energy", unit: "$/kWh", rate: 0.12 },
    { name: "Peak demand", kind: "demand", unit: "$/kW", rate: 18.5 },
  ],
  tou_heatmap: Array.from({ length: 12 }, () =>
    Array.from({ length: 24 }, (_, h) => (h >= 16 && h < 21 ? 0.32 : 0.1)),
  ),
};

const BILL = {
  bill: {
    tariff_id: "ci-tou",
    total: 412.5,
    currency: "USD",
    line_items: [
      { name: "Customer charge", kind: "fixed", amount: 75 },
      { name: "Energy", kind: "energy", amount: 250 },
      { name: "Peak demand", kind: "demand", amount: 87.5 },
    ],
  },
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
  await context.route("**/api/proxy/api/v1/tariffs", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(TARIFFS) }),
  );
  await context.route("**/api/proxy/api/v1/tariffs/ci-tou", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(TARIFF_CITOU) }),
  );
  await context.route("**/api/proxy/api/v1/tariffs/ci-tou/simulate", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(BILL) }),
  );
});

test("select tariff, run synthetic simulation, see bill breakdown", async ({
  page,
}) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("pw");
  await page.getByRole("button", { name: /sign in/i }).click();

  await page.goto("/tariffs");
  await expect(page.getByTestId("tariffs-view")).toBeVisible();

  await page.getByRole("button", { name: /C&I — TOU Demand/ }).click();
  await expect(page.getByTestId("tariff-detail")).toBeVisible();

  await page.getByRole("tab", { name: /simulate/i }).click();
  await expect(page.getByTestId("synthetic-toggle")).toBeChecked();
  await page.getByTestId("run-simulation").click();

  await expect(page.getByTestId("bill-breakdown")).toBeVisible();
  await expect(page.getByTestId("bill-total")).toHaveText("$412.50");
});
