import { test, expect } from "@playwright/test";

const FAKE_TOKEN = "fake.jwt.token";

const INITIAL_YAML = `service:
  name: vpp
  port: 8000
optimizer:
  solver: GLPK
  gap_tolerance: 0.001
`;

const SCHEMA = {
  type: "object",
  required: ["service"],
  additionalProperties: true,
  properties: {
    service: {
      type: "object",
      required: ["name", "port"],
      properties: {
        name: { type: "string" },
        port: { type: "integer", minimum: 1, maximum: 65535 },
      },
    },
    optimizer: {
      type: "object",
      properties: {
        solver: { type: "string" },
        gap_tolerance: { type: "number", minimum: 0 },
      },
    },
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
  await context.route("**/api/proxy/api/v1/config/schema", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(SCHEMA) }),
  );
  await context.route("**/api/proxy/api/v1/config", (route) => {
    if (route.request().method() === "PUT") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ yaml: INITIAL_YAML, version: 2 }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ yaml: INITIAL_YAML, version: 1 }),
    });
  });
});

test("load YAML, modify with bad value, validate shows error, fix, see diff, apply succeeds", async ({
  page,
}) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("pw");
  await page.getByRole("button", { name: /sign in/i }).click();

  await page.goto("/settings");
  await expect(page.getByTestId("settings-view")).toBeVisible();

  // Wait for Monaco to appear (it lazy-loads)
  await page.waitForSelector(".monaco-editor", { timeout: 30_000 });

  // Replace editor contents with invalid YAML (bad indent: tab in indent)
  // Easier: write invalid value (port: "not-a-number") to trigger schema error.
  const bad = `service:\n  name: vpp\n  port: "not-a-number"\n`;
  await page.evaluate((text) => {
    // monaco exposes models via window.monaco
    interface MonacoLike {
      editor: {
        getModels: () => { setValue: (s: string) => void }[];
      };
    }
    const w = window as unknown as { monaco?: MonacoLike };
    const models = w.monaco?.editor.getModels();
    if (models && models[0]) models[0].setValue(text);
  }, bad);

  await page.getByTestId("validate-button").click();
  await expect(page.getByTestId("validation-errors")).toBeVisible();

  // Fix — set a valid YAML (different from initial so diff appears).
  const ok = `service:\n  name: vpp\n  port: 9090\noptimizer:\n  solver: GLPK\n  gap_tolerance: 0.001\n`;
  await page.evaluate((text) => {
    interface MonacoLike {
      editor: { getModels: () => { setValue: (s: string) => void }[] };
    }
    const w = window as unknown as { monaco?: MonacoLike };
    const models = w.monaco?.editor.getModels();
    if (models && models[0]) models[0].setValue(text);
  }, ok);

  await page.getByTestId("validate-button").click();
  await expect(page.getByTestId("validation-errors")).toHaveCount(0);

  await page.getByTestId("diff-tab").click();
  await expect(page.getByTestId("config-diff")).toBeVisible();

  await page.getByTestId("apply-button").click();
  await expect(page.getByTestId("apply-confirm")).toBeVisible();
  await page.getByTestId("apply-confirm-button").click();
  // Toast appears on success — check by text on the document.
  await expect(page.getByText(/configuration applied/i)).toBeVisible();
});
