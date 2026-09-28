import { test, expect } from "@playwright/test";
import { api, json, signInAs } from "./support";

const PROTOCOLS = [
  {
    name: "modbus",
    version: "tcp",
    status: "connected",
    mode: "live",
    simulated: false,
    messages_sent: 10,
    messages_received: 1200,
    errors: 2,
    uptime_seconds: 7260,
  },
  {
    name: "openadr",
    version: "2.0b",
    status: "simulated",
    mode: "simulated",
    simulated: true,
    messages_sent: 0,
    messages_received: 0,
    errors: 0,
    uptime_seconds: 30,
  },
  {
    name: "ocpp",
    version: "1.6-J",
    status: "disconnected",
    mode: "live",
    simulated: false,
    messages_sent: 0,
    messages_received: 0,
    errors: 0,
    uptime_seconds: 0,
  },
];

test("operator sees live/simulated modes and can connect an adapter", async ({ context, page }) => {
  await signInAs(context, "operator");
  let connected = "";
  await context.route(api("/api/v1/protocols"), (r) => json(r, PROTOCOLS));
  await context.route(/\/api\/proxy\/api\/v1\/protocols\/[^/]+\/connect$/, (r) => {
    connected = r.request().url().split("/").slice(-2)[0];
    return json(r, { name: connected, status: "connected", message: "Connected" });
  });

  await page.goto("/protocols");
  await expect(page.getByTestId("protocols-view")).toContainText("2 live, 1 simulated");
  const modbus = page.getByTestId("protocol-modbus");
  await expect(modbus.getByTestId("mode-badge")).toHaveText(/live/i);
  await expect(modbus).toContainText("connected");
  await expect(modbus).toContainText("2h 1m");
  const openadr = page.getByTestId("protocol-openadr");
  await expect(openadr.getByTestId("mode-badge")).toHaveText(/simulated/i);
  // Running adapters offer disconnect, stopped ones connect.
  await expect(openadr.getByRole("button", { name: "Disconnect openadr" })).toBeVisible();

  await page.getByRole("button", { name: "Connect ocpp" }).click();
  await expect(page.getByText("ocpp: Connected")).toBeVisible();
  expect(connected).toBe("ocpp");
});

test("viewer sees status but no connect actions", async ({ context, page }) => {
  await signInAs(context, "viewer");
  await context.route(api("/api/v1/protocols"), (r) => json(r, PROTOCOLS));
  await page.goto("/protocols");
  await expect(page.getByTestId("protocol-ocpp")).toBeVisible();
  await expect(page.getByRole("button", { name: /connect/i })).toHaveCount(0);
});

test("explains how to enable adapters when none run", async ({ context, page }) => {
  await signInAs(context, "operator");
  await context.route(api("/api/v1/protocols"), (r) => json(r, []));
  await page.goto("/protocols");
  await expect(page.getByTestId("protocols-empty")).toContainText("VPP_OCPP_ENABLED");
});

test("sidebar links to the new operator pages", async ({ context, page }) => {
  await signInAs(context, "operator");
  await context.route(api("/api/v1/protocols"), (r) => json(r, []));
  await page.goto("/protocols");
  const nav = page.getByRole("navigation", { name: "Primary" });
  for (const name of ["Trading", "Optimization", "Dispatches", "Protocols"]) {
    await expect(nav.getByRole("link", { name })).toBeVisible();
  }
  await expect(nav.getByRole("link", { name: "Protocols" })).toHaveAttribute(
    "aria-current",
    "page",
  );
});
