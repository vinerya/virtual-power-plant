import { test, expect, type Page, type Route } from "@playwright/test";

// Every response below is shaped exactly like the FastAPI tariff API
// (src/vpp/schemas/tariffs.py): TariffRead with derived components /
// tou_heatmap, BillResponse with flat line_items[{label}] + comparison.

const FAKE_TOKEN = "fake.jwt.token";
const NOW = "2026-09-01T00:00:00Z";

type Json = Record<string, unknown>;

function heat(peak: number, off: number) {
  return Array.from({ length: 12 }, () =>
    Array.from({ length: 24 }, (_, h) => (h >= 16 && h < 21 ? peak : off)),
  );
}

function tariff(id: string, name: string, extra: Json = {}): Json {
  return {
    id,
    name,
    utility: "Pacific Gas and Electric",
    urdb_label: null,
    urdb_json: { name, sector: "Residential" },
    effective_date: "2024-03-01",
    created_at: NOW,
    updated_at: NOW,
    sector: "Residential",
    source: null,
    description: null,
    components: [
      {
        name: "Energy — period 0",
        kind: "energy",
        unit: "$/kWh",
        rate: 0.39,
        schedule: ["Jun-Sep, every day 00:00-16:00"],
      },
      {
        name: "Energy — period 1",
        kind: "energy",
        unit: "$/kWh",
        rate: 0.5,
        schedule: ["Jun-Sep, every day 16:00-21:00"],
      },
      { name: "Minimum bill", kind: "minimum", unit: "$/month", rate: 10.55 },
      { name: "Utility users tax", kind: "tax", unit: "fraction", rate: 0.015 },
    ],
    tou_heatmap: heat(0.5, 0.39),
    tou_heatmap_weekend: heat(0.46, 0.36),
    is_tou: true,
    nem_regime: "none",
    nem_source: "default",
    parse_error: null,
    ...extra,
  };
}

function bill(tariffId: string, tariffName: string, total: number, extra: Json = {}): Json {
  return {
    total,
    tariff_name: tariffName,
    tariff_id: tariffId,
    currency: "USD",
    line_items: [
      { kind: "energy", label: "TOU period_0", quantity: 400, unit: "kWh", rate: 0.39, amount: 156 },
      { kind: "energy", label: "TOU period_1", quantity: 180, unit: "kWh", rate: 0.5, amount: 90 },
      { kind: "tax", label: "Utility users tax", quantity: 246, unit: "$", rate: 0.015, amount: total - 246 },
    ],
    period_start: "2026-09-01T00:00:00-07:00",
    period_end: "2026-10-01T00:00:00-07:00",
    cycles: [
      {
        period_start: "2026-09-01T00:00:00-07:00",
        period_end: "2026-10-01T00:00:00-07:00",
        total,
        export_credit: 0,
      },
    ],
    nem_regime: "none",
    nem_source: "default",
    export_kwh: 0,
    export_credit: 0,
    notes: [],
    load_summary: {
      source: "synthetic",
      method: "deterministic residential shape, 0.8 kW average (illustrative, not metered)",
      timezone: "America/Los_Angeles",
      interval_minutes: 60,
      intervals: 720,
      import_kwh: 580.2,
      export_kwh: 0,
      peak_kw: 1.9,
    },
    comparison: null,
    ...extra,
  };
}

const PRESET = {
  id: "pge_etouc",
  name: "PG&E E-TOU-C",
  utility: "Pacific Gas and Electric",
  sector: "Residential",
  description: "PG&E E-TOU-C residential time-of-use rate.",
  source_date: "2024-03-01",
  illustrative: false,
  urdb_json: { name: "PG&E E-TOU-C", utility: "Pacific Gas and Electric", startdate: "2024-03-01" },
};

interface Backend {
  tariffs: Json[];
  requests: { method: string; path: string; body: Json | null }[];
  urdbConfigured: boolean;
  simulate?: (id: string, body: Json) => { status: number; body: Json };
}

function json(route: Route, status: number, body: unknown) {
  return route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockBackend(page: Page, role: "admin" | "viewer", backend: Backend) {
  await page.context().addCookies([
    {
      name: "vpp_session",
      value: FAKE_TOKEN,
      domain: "localhost",
      path: "/",
      httpOnly: true,
      sameSite: "Lax",
    },
  ]);
  await page.route("**/api/proxy/health", (r) => json(r, 200, { status: "ok" }));
  await page.route(/\/api\/proxy\/api\/v1\/auth\/me$/, (r) =>
    json(r, 200, { id: "u1", username: role, role, is_active: true, created_at: NOW, audience: "operator" }),
  );
  await page.route(/\/api\/proxy\/api\/v1\/tariffs(\/.*)?(\?.*)?$/, async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    const path = url.pathname.replace("/api/proxy", "");
    const method = req.method();
    const raw = req.postData();
    const body = raw ? (JSON.parse(raw) as Json) : null;
    backend.requests.push({ method, path, body });

    if (path === "/api/v1/tariffs" && method === "GET") return json(route, 200, backend.tariffs);
    if (path === "/api/v1/tariffs" && method === "POST") {
      const t = tariff(`t-${backend.tariffs.length + 1}`, String(body?.name), {
        utility: body?.utility ?? "",
        urdb_json: body?.urdb_json ?? {},
      });
      backend.tariffs.push(t);
      return json(route, 201, t);
    }
    if (path === "/api/v1/tariffs/presets")
      return json(route, 200, [{ ...PRESET, urdb_json: undefined }]);
    if (path === "/api/v1/tariffs/presets/pge_etouc") return json(route, 200, PRESET);
    if (path === "/api/v1/tariffs/import-urdb" && method === "GET")
      return json(route, 200, {
        configured: backend.urdbConfigured,
        detail: backend.urdbConfigured
          ? "OpenEI URDB import is available"
          : "Set OPENEI_API_KEY on the API server to import tariffs from OpenEI URDB",
      });
    if (path === "/api/v1/tariffs/import-urdb" && method === "POST") {
      const t = tariff("t-urdb", "Imported URDB rate", {
        urdb_label: body?.urdb_label,
        source: "URDB",
      });
      backend.tariffs.push(t);
      return json(route, 201, t);
    }
    const sim = path.match(/^\/api\/v1\/tariffs\/([^/]+)\/simulate$/);
    if (sim && method === "POST") {
      const out = backend.simulate
        ? backend.simulate(sim[1], body ?? {})
        : { status: 200, body: bill(sim[1], "PG&E E-TOU-C", 250) };
      return json(route, out.status, out.body);
    }
    const one = path.match(/^\/api\/v1\/tariffs\/([^/]+)$/);
    if (one) {
      const idx = backend.tariffs.findIndex((t) => t.id === one[1]);
      if (idx < 0) return json(route, 404, { detail: "Tariff not found" });
      if (method === "GET") return json(route, 200, backend.tariffs[idx]);
      if (method === "PUT") {
        backend.tariffs[idx] = { ...backend.tariffs[idx], ...body };
        return json(route, 200, backend.tariffs[idx]);
      }
      if (method === "DELETE") {
        backend.tariffs.splice(idx, 1);
        return route.fulfill({ status: 204 });
      }
    }
    return json(route, 404, { detail: `unmocked ${method} ${path}` });
  });
}

function freshBackend(extra: Partial<Backend> = {}): Backend {
  return {
    tariffs: [
      tariff("etouc", "PG&E E-TOU-C"),
      tariff("tou-d", "SCE TOU-D-PRIME", {
        utility: "Southern California Edison",
        nem_regime: "nem2",
        nem_source: "urdb_dgrules",
      }),
    ],
    requests: [],
    urdbConfigured: false,
    ...extra,
  };
}

test("viewer: browse schedule & components, run synthetic simulation with comparison", async ({
  page,
}) => {
  const backend = freshBackend({
    simulate: (id, body) => ({
      status: 200,
      body: bill(id, "PG&E E-TOU-C", 250, {
        comparison: body.compare_to
          ? bill(String(body.compare_to), "SCE TOU-D-PRIME", 231.4)
          : null,
      }),
    }),
  });
  await mockBackend(page, "viewer", backend);

  await page.goto("/tariffs");
  await expect(page.getByTestId("tariffs-view")).toBeVisible();
  // Viewers get no admin actions.
  await expect(page.getByRole("button", { name: "New tariff" })).toHaveCount(0);

  await page.getByRole("button", { name: /PG&E E-TOU-C/ }).click();
  const detail = page.getByTestId("tariff-detail");
  await expect(detail).toBeVisible();
  await expect(page).toHaveURL(/\/tariffs\/etouc$/);
  await expect(detail.getByRole("heading", { name: "PG&E E-TOU-C" })).toBeVisible();
  await expect(page.getByTestId("edit-tariff")).toHaveCount(0);

  // Schedule heatmap from the backend's tou_heatmap, weekday/weekend toggle.
  await expect(page.getByTestId("schedule-heatmap")).toBeVisible();
  await expect(page.getByRole("gridcell", { name: "Jul 17:00, 0.500 $/kWh" })).toBeVisible();
  await page.getByRole("button", { name: "Weekends & holidays" }).click();
  await expect(page.getByRole("gridcell", { name: "Jul 17:00, 0.460 $/kWh" })).toBeVisible();

  await page.getByRole("tab", { name: "Components" }).click();
  const list = page.getByTestId("components-list");
  await expect(list.getByText("Energy — period 1")).toBeVisible();
  await expect(list.getByText("Jun-Sep, every day 16:00-21:00")).toBeVisible();
  await expect(list.getByText("1.50%")).toBeVisible();

  await page.getByRole("tab", { name: /simulate/i }).click();
  await expect(page.getByTestId("synthetic-toggle")).toBeChecked();
  await page.getByLabel("Timezone (TOU hours)").selectOption("America/Los_Angeles");
  await page.getByLabel("Rooftop PV (kW peak)").fill("5");
  await page.getByLabel("Compare to (optional)").selectOption("tou-d");
  await page.getByTestId("run-simulation").click();

  await expect(page.getByTestId("simulation-result")).toBeVisible();
  const breakdowns = page.getByTestId("bill-breakdown");
  await expect(breakdowns).toHaveCount(2);
  await expect(page.getByTestId("bill-total").first()).toHaveText("$250.00");
  await expect(page.getByTestId("bill-total").nth(1)).toHaveText("$231.40");
  await expect(breakdowns.first().getByText("TOU period_1")).toBeVisible();
  await expect(breakdowns.first().getByText("vs SCE TOU-D-PRIME: +$18.60")).toBeVisible();
  await expect(page.getByTestId("load-summary")).toContainText("Synthetic");
  await expect(page.getByTestId("load-summary")).toContainText("illustrative, not metered");

  const simReq = backend.requests.find((r) => r.path.endsWith("/etouc/simulate"));
  expect(simReq?.body).toMatchObject({
    synthetic: { pv_kw: 5 },
    timezone: "America/Los_Angeles",
    compare_to: "tou-d",
    period_days: 30,
  });
  expect(simReq?.body).not.toHaveProperty("meter_trace");
  expect(simReq?.body).not.toHaveProperty("csv");
});

test("viewer: CSV upload simulation and server-side error display", async ({ page }) => {
  let calls = 0;
  const backend = freshBackend({
    simulate: (id, body) => {
      calls += 1;
      if (calls === 1)
        return { status: 400, body: { detail: "Invalid CSV: row 3: cannot parse timestamp" } };
      return {
        status: 200,
        body: bill(id, "PG&E E-TOU-C", 12.34, {
          load_summary: {
            source: "csv",
            method: null,
            timezone: String(body.timezone),
            interval_minutes: 15,
            intervals: 96,
            import_kwh: 24,
            export_kwh: 3,
            peak_kw: 2,
          },
        }),
      };
    },
  });
  await mockBackend(page, "viewer", backend);

  await page.goto("/tariffs/etouc");
  await expect(page.getByTestId("tariff-detail")).toBeVisible();
  await page.getByRole("tab", { name: /simulate/i }).click();
  await page.getByTestId("csv-toggle").check();
  await expect(page.getByTestId("run-simulation")).toBeDisabled();
  await page.getByTestId("csv-upload").setInputFiles({
    name: "meter.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("timestamp,kw\n2026-09-01T00:00,1.5\n2026-09-01T00:15,-0.5\n"),
  });
  await expect(page.getByText("meter.csv")).toBeVisible();
  await page.getByTestId("run-simulation").click();
  await expect(page.getByTestId("simulation-error")).toHaveText(
    "Invalid CSV: row 3: cannot parse timestamp",
  );

  await page.getByTestId("run-simulation").click();
  await expect(page.getByTestId("bill-total")).toHaveText("$12.34");
  await expect(page.getByTestId("load-summary")).toContainText("CSV upload");
  const req = backend.requests.filter((r) => r.path.endsWith("/simulate")).at(-1);
  expect(String(req?.body?.csv)).toContain("timestamp,kw");
  expect(req?.body).not.toHaveProperty("synthetic");
});

test("admin: create from preset, edit, delete", async ({ page }) => {
  const backend = freshBackend();
  await mockBackend(page, "admin", backend);

  await page.goto("/tariffs");
  await page.getByRole("button", { name: "New tariff" }).click();
  const editor = page.getByTestId("tariff-editor");
  await expect(editor).toBeVisible();
  await editor.getByLabel("Start from preset").selectOption("pge_etouc");
  await expect(editor.getByLabel("Name")).toHaveValue("PG&E E-TOU-C");
  await editor.getByLabel("Name").fill("PG&E E-TOU-C (ops copy)");
  await editor.getByLabel("Export credit (NEM)").selectOption("nem2");
  await editor.getByRole("button", { name: "Create tariff" }).click();

  const post = backend.requests.find((r) => r.method === "POST" && r.path === "/api/v1/tariffs");
  expect(post?.body).toMatchObject({
    name: "PG&E E-TOU-C (ops copy)",
    utility: "Pacific Gas and Electric",
    effective_date: "2024-03-01",
    urdb_json: { name: "PG&E E-TOU-C", nem: "nem2" },
  });
  const detail = page.getByTestId("tariff-detail");
  await expect(detail.getByRole("heading", { name: "PG&E E-TOU-C (ops copy)" })).toBeVisible();
  await expect(page).toHaveURL(/\/tariffs\/t-3$/);

  // Edit.
  await page.getByTestId("edit-tariff").click();
  await expect(editor.getByLabel("Name")).toHaveValue("PG&E E-TOU-C (ops copy)");
  await editor.getByLabel("Name").fill("Renamed tariff");
  await editor.getByRole("button", { name: "Save changes" }).click();
  await expect(detail.getByRole("heading", { name: "Renamed tariff" })).toBeVisible();
  const put = backend.requests.find((r) => r.method === "PUT");
  expect(put?.path).toBe("/api/v1/tariffs/t-3");
  expect(put?.body).toMatchObject({ name: "Renamed tariff" });

  // Invalid JSON is caught client-side.
  await page.getByTestId("edit-tariff").click();
  await editor.getByLabel("URDB JSON").fill("{ not json");
  await editor.getByRole("button", { name: "Save changes" }).click();
  await expect(editor.getByRole("alert")).toBeVisible();
  await page.keyboard.press("Escape");

  // Delete with confirmation.
  await page.getByTestId("delete-tariff").click();
  await page.getByTestId("confirm-delete-tariff").click();
  await expect(page.getByText("Select a tariff from the list")).toBeVisible();
  expect(backend.requests.some((r) => r.method === "DELETE" && r.path === "/api/v1/tariffs/t-3")).toBe(true);
  await expect(page.getByRole("button", { name: /Renamed tariff/ })).toHaveCount(0);
});

test("admin: URDB import explains a missing OPENEI_API_KEY, imports when configured", async ({
  page,
}) => {
  const backend = freshBackend({ urdbConfigured: false });
  await mockBackend(page, "admin", backend);

  await page.goto("/tariffs");
  await page.getByRole("button", { name: "Import from URDB" }).click();
  await expect(page.getByTestId("urdb-not-configured")).toContainText("OPENEI_API_KEY");
  await expect(page.getByLabel("URDB record id")).toBeDisabled();
  await page.keyboard.press("Escape");

  backend.urdbConfigured = true;
  await page.reload();
  await page.getByRole("button", { name: "Import from URDB" }).click();
  await expect(page.getByLabel("URDB record id")).toBeEnabled();
  await page.getByLabel("URDB record id").fill("5b3104b95457a3f7437a9b2d");
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(
    page.getByTestId("tariff-detail").getByRole("heading", { name: "Imported URDB rate" }),
  ).toBeVisible();
  const post = backend.requests.find(
    (r) => r.method === "POST" && r.path === "/api/v1/tariffs/import-urdb",
  );
  expect(post?.body).toMatchObject({ urdb_label: "5b3104b95457a3f7437a9b2d" });
});
