import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";

const ME = {
  id: "c-1",
  name: "Demo Member",
  address: "1 Sunlit Lane",
  tariff_id: "flat-residential",
  baseline_kwh_per_month: 850,
};

const BILL = {
  bill: {
    total: 142.7,
    currency: "USD",
    line_items: [
      { kind: "fixed", name: "Customer charge", amount: 12 },
      { kind: "energy", name: "Energy (820 kWh)", amount: 130.7 },
    ],
  },
  baseline: {
    total: 178.4,
    currency: "USD",
    line_items: [
      { kind: "fixed", name: "Customer charge", amount: 12 },
      { kind: "energy", name: "Energy (1020 kWh)", amount: 166.4 },
    ],
  },
  this_month_kwh: 820,
  last_month_kwh: 905,
};

const PROGRAMS = [
  {
    id: "summer-peak",
    name: "Summer Peak Saver",
    description: "Save energy during summer peak",
    incentive_per_event: 25,
  },
  {
    id: "always-on",
    name: "Always-on Battery",
    description: "Battery available year round",
    incentive_per_event: 12,
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

  await context.route(
    /\/api\/proxy\/api\/v1\/customer\/me$/,
    (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(ME),
      }),
  );
  await context.route(
    /\/api\/proxy\/api\/v1\/customer\/me\/bill(\?.*)?$/,
    (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(BILL),
      }),
  );
  await context.route(
    /\/api\/proxy\/api\/v1\/customer\/me\/devices$/,
    (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([]),
      }),
  );
  await context.route(
    /\/api\/proxy\/api\/v1\/customer\/programs$/,
    (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(PROGRAMS),
      }),
  );
  await context.route(
    /\/api\/proxy\/api\/v1\/customer\/enrollments$/,
    (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ok: true, enrolled: ["summer-peak"] }),
      }),
  );
});

test("portal: see savings, navigate bill & enrollment, opt in", async ({
  page,
}) => {
  await page.goto("/portal");
  await expect(page.getByTestId("savings-card")).toBeVisible();
  await expect(page.getByTestId("savings-amount")).toContainText("$");

  // Navigate to bill page.
  await page.goto("/portal/bill");
  await expect(page.getByTestId("customer-bill-page")).toBeVisible();
  await expect(page.getByTestId("bill-total").first()).toContainText("142.70");

  // Navigate to enrollment.
  await page.goto("/portal/enroll");
  await expect(page.getByTestId("enrollment-form")).toBeVisible();
  const cards = page.getByTestId("program-card");
  await expect(cards).toHaveCount(2);

  // Opt into a program.
  await cards.first().click();
  await page.getByTestId("ack-checkbox").check();
  await page.getByTestId("enroll-submit").click();
  await expect(page.getByTestId("enrollment-success")).toBeVisible();
});

test("portal: energy flow is derived from live device readings", async ({ page }) => {
  await page.route(/\/api\/proxy\/api\/v1\/customer\/me\/devices$/, (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        { id: "pv", kind: "solar", name: "Roof", state: "generating", current_power: 4.2, online: true, rated_power: 6 },
        { id: "bat", kind: "battery", name: "Wall", state: "charging", current_power: 1.5, state_of_charge: 0.6, online: true, rated_power: 5 },
        { id: "ev", kind: "ev_charger", name: "EV", state: "consuming", current_power: 2.0, online: true, rated_power: 7 },
        { id: "off", kind: "solar", name: "Shed", state: "offline", current_power: 9, online: false, rated_power: 2 },
      ]),
    }),
  );
  await page.goto("/portal");
  const flow = page.getByTestId("energy-flow");
  await expect(flow.getByTestId("flow-generation")).toContainText("4.2 kW");
  await expect(flow.getByTestId("flow-battery")).toContainText("1.5 kW");
  await expect(flow.getByTestId("flow-battery")).toContainText("charging");
  await expect(flow.getByTestId("flow-battery")).toContainText("60.0% full");
  await expect(flow.getByTestId("flow-load")).toContainText("2.0 kW");
  // 4.2 generated - 1.5 into the battery - 2.0 EV = 0.7 kW net supply.
  await expect(flow.getByTestId("flow-net")).toContainText("0.7 kW");
  await expect(flow).toContainText("1 offline, not counted");
  // The old hard-coded demo numbers are gone.
  await expect(flow).not.toContainText("2.4 kW");
});

test("portal: a 409 bill shows the server's reason, not a generic failure", async ({ page }) => {
  await page.route(/\/api\/proxy\/api\/v1\/customer\/me\/bill(\?.*)?$/, (r) =>
    r.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({ detail: "No tariff is assigned to this account yet" }),
    }),
  );
  await page.goto("/portal/bill");
  await expect(page.getByTestId("bill-unavailable-detail")).toHaveText(
    "No tariff is assigned to this account yet",
  );
  await expect(page.getByText("Failed to load your bill.")).toHaveCount(0);

  // The overview still renders (with the same explanation) instead of failing.
  await page.goto("/portal");
  await expect(page.getByTestId("bill-unavailable")).toBeVisible();
  await expect(page.getByTestId("energy-flow")).toBeVisible();
});
