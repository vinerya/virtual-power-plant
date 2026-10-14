import { test, expect } from "@playwright/test";
import { api, json, signInAs } from "./support";

const PROTOCOLS = [
  {
    name: "openadr",
    version: "2.0b",
    status: "connected",
    mode: "live",
    simulated: false,
    messages_sent: 12,
    messages_received: 12,
    errors: 0,
    uptime_seconds: 90,
  },
];

const NOW = Math.floor(Date.now() / 1000);

const EVENTS = {
  protocol: "openadr",
  status: "connected",
  mode: "live",
  simulated: false,
  ven_id: "ven_vpp",
  vtn_id: "openleadr-vtn",
  registered: true,
  events: [
    {
      event_id: "evt-simple-1",
      signal_type: "SIMPLE",
      signal_level: 1,
      start_time: NOW + 60,
      duration_seconds: 600,
      end_time: NOW + 660,
      status: "pending",
      active: false,
      current_value: 1,
      opt_type: "optIn",
      opted_at: NOW - 5,
      source: "vtn",
      market_context: "oadr://unknown.context",
      modification_number: 0,
      priority: 0,
      test_event: false,
      response_required: "always",
      vtn_status: "far",
      signals: [
        {
          signal_name: "SIMPLE",
          signal_type: "level",
          signal_id: "sig-1",
          current_value: 0,
          intervals: [{ uid: "0", start_time: NOW + 60, duration_seconds: 600, value: 1 }],
        },
      ],
    },
    {
      event_id: "evt-cancelled-2",
      signal_type: "LOAD_DISPATCH",
      signal_level: 30,
      start_time: NOW - 900,
      duration_seconds: 300,
      status: "cancelled",
      active: false,
      opt_type: "optOut",
      source: "vtn",
      signals: [],
    },
  ],
};

const RESPONSES = [
  {
    id: 7,
    created_at: NOW - 4,
    protocol: "openadr",
    source_id: "evt-simple-1",
    revision: 0,
    action: "opt",
    opt_type: "optIn",
    signal_type: "SIMPLE",
    signal_level: 1,
    target_kw: 25,
    delivered_kw: null,
    window_start: NOW + 60,
    window_end: NOW + 660,
    run_id: null,
    reason: "fleet covers request",
    details: {},
  },
];

const DR_STATUS = {
  running: true,
  auto_response_enabled: false,
  active: null,
  dispatch_count: 0,
  last_tick_at: NOW,
  last_error: null,
  protocols: { openadr: true, ieee2030_5: false, ocpp: false },
  device_control_enabled: false,
};

async function stubBackend(context: Parameters<typeof signInAs>[0]) {
  await context.route(api("/api/v1/protocols"), (r) => json(r, PROTOCOLS));
  await context.route(api("/api/v1/protocols/openadr/events"), (r) => json(r, EVENTS));
  await context.route(api("/api/v1/dr/responses"), (r) => json(r, RESPONSES));
  await context.route(api("/api/v1/dr/status"), (r) => json(r, DR_STATUS));
}

test("operator sees DR events, signals and the decision log, and can override opt", async ({
  context,
  page,
}) => {
  await signInAs(context, "operator");
  await stubBackend(context);
  let override: { eventId: string; body: unknown } | null = null;
  await context.route(/\/api\/proxy\/api\/v1\/protocols\/openadr\/events\/[^/]+\/opt$/, (r) => {
    const eventId = r.request().url().split("/").slice(-2)[0];
    override = { eventId, body: r.request().postDataJSON() };
    return json(r, {
      protocol: "openadr",
      status: "connected",
      mode: "live",
      simulated: false,
      event_id: eventId,
      opt_type: "optOut",
      sent_to_vtn: true,
      released_dispatch: false,
    });
  });

  await page.goto("/protocols");
  const panel = page.getByTestId("protocol-messages");
  await expect(panel).toBeVisible();
  await expect(panel.getByTestId("dr-summary")).toContainText("registered");
  await expect(panel.getByTestId("dr-summary")).toContainText("openleadr-vtn");
  await expect(panel.getByTestId("dr-summary")).toContainText("auto-response off");

  const table = panel.getByTestId("dr-events-table");
  const row = panel.getByTestId("dr-event-evt-simple-1");
  await expect(row).toContainText("SIMPLE");
  await expect(row).toContainText("optIn");
  await expect(row).toContainText("pending");
  await expect(panel.getByTestId("dr-event-evt-cancelled-2")).toContainText("cancelled");
  // Final events cannot be overridden.
  await expect(
    panel.getByRole("button", { name: "Opt out of event evt-cancelled-2" }),
  ).toBeDisabled();

  await row.getByRole("button", { name: /Show signals for event evt-simple-1/ }).click();
  await expect(panel.getByTestId("dr-event-evt-simple-1-signals")).toContainText("SIMPLE/level");
  await expect(panel.getByTestId("dr-event-evt-simple-1-signals")).toContainText(
    "oadr://unknown.context",
  );

  await panel.getByRole("button", { name: "Opt out of event evt-simple-1" }).click();
  await expect(page.getByText("Sent to the VTN.")).toBeVisible();
  expect(override).not.toBeNull();
  expect(override!.eventId).toBe("evt-simple-1");
  expect(override!.body).toMatchObject({ opt_type: "optOut" });
  await expect(table).toBeVisible();

  await panel.getByRole("tab", { name: "Decision log" }).click();
  const log = panel.getByTestId("dr-log-table");
  await expect(log).toContainText("opt");
  await expect(log).toContainText("25.0 kW");
  await expect(log).toContainText("fleet covers request");

  await panel.getByRole("tab", { name: /Live feed/ }).click();
  await expect(panel.getByTestId("live-feed-empty")).toBeVisible();
});

test("viewer sees messages but no override buttons", async ({ context, page }) => {
  await signInAs(context, "viewer");
  await stubBackend(context);
  await page.goto("/protocols");
  const panel = page.getByTestId("protocol-messages");
  await expect(panel.getByTestId("dr-event-evt-simple-1")).toBeVisible();
  await expect(panel.getByRole("button", { name: /Opt (in|out)/ })).toHaveCount(0);
});

test("panel is hidden when no DR-capable adapter runs", async ({ context, page }) => {
  await signInAs(context, "operator");
  await context.route(api("/api/v1/protocols"), (r) =>
    json(r, [{ ...PROTOCOLS[0], name: "modbus", version: "tcp" }]),
  );
  await page.goto("/protocols");
  await expect(page.getByTestId("protocol-modbus")).toBeVisible();
  await expect(page.getByTestId("protocol-messages")).toHaveCount(0);
});
