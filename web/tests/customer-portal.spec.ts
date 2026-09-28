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
