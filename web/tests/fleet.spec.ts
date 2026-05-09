import { test, expect } from "@playwright/test";

/**
 * Smoke test for M1.
 *
 * The Next.js routes (/api/auth/login and /api/proxy/*) are mocked at the
 * network layer so this test can run without the FastAPI backend.
 */

const FAKE_TOKEN = "fake.jwt.token";

const RESOURCES = [
  {
    id: "r1",
    name: "Battery A",
    resource_type: "battery",
    rated_power: 250,
    metadata: {},
    online: true,
    current_power: 120,
    efficiency: 0.95,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: "r2",
    name: "Solar Field",
    resource_type: "solar",
    rated_power: 500,
    metadata: {},
    online: true,
    current_power: 320,
    efficiency: 0.92,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
];

test.beforeEach(async ({ context }) => {
  await context.route("**/api/auth/login", async (route) => {
    const body = JSON.parse(route.request().postData() || "{}");
    if (body.username === "operator" && body.password === "correct-pass") {
      await route.fulfill({
        status: 200,
        headers: {
          "set-cookie": `vpp_session=${FAKE_TOKEN}; Path=/; HttpOnly; SameSite=Lax`,
        },
        contentType: "application/json",
        body: JSON.stringify({ ok: true }),
      });
    } else {
      await route.fulfill({
        status: 401,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Invalid credentials" }),
      });
    }
  });

  await context.route("**/api/proxy/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    });
  });

  await context.route("**/api/proxy/api/v1/resources/", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(RESOURCES),
    });
  });
});

test("rejects bad credentials", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("wrong");
  await page.getByRole("button", { name: /sign in/i }).click();
  await expect(page.getByTestId("login-error")).toBeVisible();
});

test("logs in and renders fleet overview", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("operator");
  await page.getByLabel("Password").fill("correct-pass");
  await page.getByRole("button", { name: /sign in/i }).click();

  await expect(page).toHaveURL("/");
  await expect(page.getByRole("heading", { name: /fleet overview/i })).toBeVisible();
  await expect(page.getByTestId("stat-total")).toHaveText("2");
  await expect(page.getByText("Battery A")).toBeVisible();
  await expect(page.getByText("Solar Field")).toBeVisible();
});
