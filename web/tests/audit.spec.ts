import { test, expect } from "@playwright/test";
import { api, signInAs } from "./support";

const NOW = new Date().toISOString();

function entry(id: number, action: string, outcome = "success") {
  return {
    id,
    ts: NOW,
    actor_id: "u-admin",
    actor_username: "admin-user",
    action,
    target_type: "user",
    target_id: `u-${id}`,
    client_ip: "203.0.113.7",
    outcome,
    details: { username: `user${id}` },
  };
}

test.describe("Settings → Audit log (admin)", () => {
  test.beforeEach(async ({ context }) => {
    await signInAs(context, "admin");
  });

  test("lists entries with filters and pages through the total", async ({ page }) => {
    const urls: string[] = [];
    await page.route(api("/api/v1/audit"), (route) => {
      const url = new URL(route.request().url());
      urls.push(url.search);
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const failuresOnly = url.searchParams.get("outcome") === "failure";
      const all = failuresOnly
        ? [entry(900, "auth.login", "failure")]
        : Array.from({ length: 60 }, (_, i) => entry(i + 1, "user.delete"));
      const items = all.slice(offset, offset + 50);
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        headers: { "x-total-count": String(all.length) },
        body: JSON.stringify(items),
      });
    });

    await page.goto("/settings/audit");
    const nav = page.getByTestId("settings-nav");
    await expect(nav.getByRole("link", { name: "Audit log" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    await expect(page.getByTestId("audit-row-1")).toBeVisible();
    await expect(page.getByTestId("audit-range")).toHaveText("1-50 of 60");

    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByTestId("audit-range")).toHaveText("51-60 of 60");
    await expect(page.getByTestId("audit-row-51")).toBeVisible();
    await expect(page.getByRole("button", { name: "Next" })).toBeDisabled();

    await page.getByLabel("Outcome").selectOption("failure");
    await expect(page.getByTestId("audit-row-900")).toBeVisible();
    await expect(page.getByTestId("audit-range")).toHaveText("1-1 of 1");
    expect(urls.at(-1)).toContain("outcome=failure");
    expect(urls.at(-1)).toContain("offset=0");
  });
});

test("non-admins get an explanation instead of the audit log", async ({ page, context }) => {
  await signInAs(context, "operator");
  await page.goto("/settings/audit");
  await expect(page.getByTestId("audit-admin-only")).toBeVisible();
  await expect(
    page.getByTestId("settings-nav").getByRole("link", { name: "Audit log" }),
  ).toHaveCount(0);
});
