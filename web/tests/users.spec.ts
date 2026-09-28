import { test, expect, type Page, type Request } from "@playwright/test";
import { api, json, signInAs } from "./support";

const NOW = new Date().toISOString();

const USERS = [
  {
    id: "u-admin",
    username: "admin-user",
    role: "admin",
    is_active: true,
    created_at: NOW,
    last_login_at: NOW,
    api_key_count: 0,
  },
  {
    id: "u-bob",
    username: "bob",
    role: "operator",
    is_active: true,
    created_at: NOW,
    last_login_at: null,
    api_key_count: 1,
  },
];

const BOB_KEY = {
  id: "k-1",
  name: "scada-bridge",
  role: "operator",
  is_active: true,
  created_at: NOW,
  last_used_at: null,
  key_prefix: "vpp_AbCdEfGh",
  user_id: "u-bob",
  username: "bob",
};

async function stubUsersApi(page: Page) {
  const calls: { method: string; url: string; body: unknown }[] = [];
  const record = (r: Request) =>
    calls.push({ method: r.method(), url: r.url(), body: r.postDataJSON?.() ?? null });
  let users = USERS.map((u) => ({ ...u }));
  let keys = [{ ...BOB_KEY }];

  await page.route(api("/api/v1/users"), (route) => {
    const req = route.request();
    if (req.method() === "POST") {
      record(req);
      const body = req.postDataJSON();
      if (String(body.password).length < 12) {
        return json(route, { detail: "Password must be at least 12 characters long" }, 422);
      }
      const created = {
        id: "u-new",
        username: body.username,
        role: body.role,
        is_active: true,
        created_at: NOW,
        last_login_at: null,
        api_key_count: 0,
      };
      users = [...users, created];
      return json(route, created, 201);
    }
    return json(route, users);
  });
  await page.route(/\/api\/proxy\/api\/v1\/users\/[^/]+$/, (route) => {
    const req = route.request();
    record(req);
    const id = req.url().split("/").pop()!;
    if (req.method() === "DELETE") {
      users = users.filter((u) => u.id !== id);
      keys = keys.filter((k) => k.user_id !== id);
      return route.fulfill({ status: 204, body: "" });
    }
    const patch = req.postDataJSON();
    users = users.map((u) =>
      u.id === id
        ? {
            ...u,
            ...patch,
            api_key_count: patch.is_active === false ? 0 : u.api_key_count,
          }
        : u,
    );
    if (patch.is_active === false) keys = keys.filter((k) => k.user_id !== id);
    return json(route, users.find((u) => u.id === id));
  });
  await page.route(/\/api\/proxy\/api\/v1\/users\/[^/]+\/(password|revoke-sessions)$/, (route) => {
    record(route.request());
    return route.fulfill({ status: 204, body: "" });
  });
  await page.route(api("/api/v1/auth/api-keys"), (route) => json(route, keys));
  await page.route(/\/api\/proxy\/api\/v1\/auth\/api-keys\/[^/?]+$/, (route) => {
    const req = route.request();
    record(req);
    const id = req.url().split("/").pop();
    keys = keys.filter((k) => k.id !== id);
    return route.fulfill({ status: 204, body: "" });
  });
  return calls;
}

test.describe("Settings → Users & API keys (admin)", () => {
  test.beforeEach(async ({ context }) => {
    await signInAs(context, "admin");
  });

  test("lists users and keys; changes roles, deactivates, resets passwords, revokes keys", async ({
    page,
  }) => {
    const calls = await stubUsersApi(page);
    await page.goto("/settings/users");
    await expect(page.getByTestId("users-view")).toBeVisible();

    const nav = page.getByTestId("settings-nav");
    await expect(nav.getByRole("link", { name: "Users & API keys" })).toHaveAttribute(
      "aria-current",
      "page",
    );

    // Own row: cannot demote or deactivate yourself.
    const self = page.getByTestId("user-row-admin-user");
    await expect(self.getByText("you")).toBeVisible();
    await expect(self.getByLabel("Role of admin-user")).toBeDisabled();
    await expect(self.getByRole("button", { name: "Deactivate admin-user" })).toBeDisabled();

    // Role change.
    const bob = page.getByTestId("user-row-bob");
    await bob.getByLabel("Role of bob").selectOption("viewer");
    await expect(page.getByText(/bob is now viewer/)).toBeVisible();
    expect(calls.find((c) => c.method === "PATCH")?.body).toEqual({ role: "viewer" });

    // Password reset.
    await bob.getByRole("button", { name: "Reset password of bob" }).click();
    await page.getByLabel("New password for bob").fill("Turbine-Harbor-9051");
    await page.getByRole("button", { name: "Set password", exact: true }).click();
    await expect(page.getByText(/Password of bob reset/)).toBeVisible();
    expect(calls.find((c) => c.url.endsWith("/u-bob/password"))?.body).toEqual({
      new_password: "Turbine-Harbor-9051",
    });

    // Revoke sessions.
    await bob.getByRole("button", { name: "Sign out bob everywhere" }).click();
    await expect(page.getByText("All sessions of bob revoked")).toBeVisible();

    // Revoke an API key (inline confirm).
    const keysCard = page.getByTestId("all-api-keys");
    await expect(keysCard.getByText("vpp_AbCdEfGh…")).toBeVisible();
    await keysCard.getByRole("button", { name: "Revoke API key scada-bridge" }).click();
    await keysCard.getByRole("button", { name: "Confirm revoke" }).click();
    await expect(page.getByText(/API key “scada-bridge” revoked/)).toBeVisible();
    await expect(keysCard.getByText("No active API keys.")).toBeVisible();
    expect(calls.some((c) => c.method === "DELETE" && c.url.endsWith("/api-keys/k-1"))).toBe(true);

    // Deactivate with confirmation.
    await bob.getByRole("button", { name: "Deactivate bob" }).click();
    await bob.getByRole("button", { name: "Confirm deactivate" }).click();
    await expect(page.getByText(/bob deactivated; sessions and API keys revoked/)).toBeVisible();
    await expect(bob.getByText("inactive")).toBeVisible();
    await expect(bob.getByRole("button", { name: "Activate bob" })).toBeVisible();

    // Delete with confirmation; not offered for your own account.
    await expect(self.getByRole("button", { name: "Delete admin-user" })).toHaveCount(0);
    await bob.getByRole("button", { name: "Delete bob" }).click();
    await bob.getByRole("button", { name: "Confirm delete" }).click();
    await expect(page.getByText(/bob deleted; their API keys were removed/)).toBeVisible();
    await expect(page.getByTestId("user-row-bob")).toHaveCount(0);
    expect(calls.some((c) => c.method === "DELETE" && c.url.endsWith("/api/v1/users/u-bob"))).toBe(
      true,
    );
  });

  test("create user shows the server's password-policy error, then succeeds", async ({ page }) => {
    const calls = await stubUsersApi(page);
    await page.goto("/settings/users");
    const form = page.getByTestId("create-user");
    await form.getByLabel("New username").fill("carol");
    await form.getByLabel("Initial password").fill("short");
    // Bypass the browser's minLength check to exercise the API error path.
    await form.getByLabel("Initial password").evaluate((el) => el.removeAttribute("minlength"));
    await form.getByLabel("Role").selectOption("operator");
    await form.getByRole("button", { name: "Create user" }).click();
    await expect(page.getByTestId("create-user-error")).toHaveText(
      "Password must be at least 12 characters long",
    );

    await form.getByLabel("Initial password").fill("Grid-Battery-4217");
    await form.getByRole("button", { name: "Create user" }).click();
    await expect(page.getByText("User carol created")).toBeVisible();
    await expect(page.getByTestId("user-row-carol")).toBeVisible();
    expect(calls.filter((c) => c.method === "POST" && c.url.endsWith("/api/v1/users")).at(-1)?.body)
      .toEqual({ username: "carol", password: "Grid-Battery-4217", role: "operator" });
  });
});

test("non-admins see no Users tab and an explanation instead of the admin page", async ({
  page,
  context,
}) => {
  await signInAs(context, "operator");
  await page.route(api("/api/v1/auth/api-keys"), (route) => json(route, []));
  await page.goto("/settings/users");
  await expect(page.getByTestId("users-admin-only")).toBeVisible();
  const nav = page.getByTestId("settings-nav");
  await expect(nav.getByRole("link", { name: "Account" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Users & API keys" })).toHaveCount(0);
});

test.describe("Settings → Account (self-service)", () => {
  test.beforeEach(async ({ context }) => {
    await signInAs(context, "operator");
  });

  test("creates an API key that is shown exactly once, lists and revokes it", async ({ page }) => {
    let keys: (typeof BOB_KEY)[] = [];
    let created: unknown = null;
    await page.route(api("/api/v1/auth/api-keys"), (route) => {
      const req = route.request();
      if (req.method() === "POST") {
        created = req.postDataJSON();
        keys = [{ ...BOB_KEY, id: "k-9", name: "ci-runner", user_id: "u-op", username: "operator-user" }];
        return json(
          route,
          {
            id: "k-9",
            name: "ci-runner",
            key: "vpp_SECRET-shown-once",
            role: "operator",
            created_at: NOW,
            key_prefix: "vpp_SECRET-s",
          },
          201,
        );
      }
      return json(route, keys);
    });
    await page.route(/\/api\/proxy\/api\/v1\/auth\/api-keys\/k-9$/, (route) => {
      keys = [];
      return route.fulfill({ status: 204, body: "" });
    });

    await page.goto("/settings/account");
    const card = page.getByTestId("my-api-keys");
    await expect(card.getByText("No active API keys.")).toBeVisible();
    // Operators may only mint keys with their own role.
    await expect(card.getByLabel("Key role").locator("option")).toHaveText(["operator"]);
    await card.getByLabel("Key name").fill("ci-runner");
    await card.getByRole("button", { name: "Create key" }).click();

    await expect(page.getByTestId("new-api-key-value")).toHaveText("vpp_SECRET-shown-once");
    expect(created).toEqual({ name: "ci-runner", role: "operator" });
    await expect(card.getByTestId("api-key-table")).toContainText("ci-runner");
    await card.getByRole("button", { name: "I have stored it" }).click();
    await expect(page.getByTestId("new-api-key")).toHaveCount(0);
    await expect(page.getByText("vpp_SECRET-shown-once")).toHaveCount(0);

    await card.getByRole("button", { name: "Revoke API key ci-runner" }).click();
    await card.getByRole("button", { name: "Confirm revoke" }).click();
    await expect(card.getByText("No active API keys.")).toBeVisible();
  });

  test("changes the password through the session route", async ({ page }) => {
    await page.route(api("/api/v1/auth/api-keys"), (route) => json(route, []));
    const bodies: unknown[] = [];
    await page.route("**/api/auth/password", (route) => {
      const body = route.request().postDataJSON();
      bodies.push(body);
      if (body.current_password !== "Old-Passphrase-77") {
        return json(route, { detail: "Current password is incorrect" }, 400);
      }
      return json(route, { ok: true });
    });
    await page.goto("/settings/account");
    const card = page.getByTestId("change-password");
    await card.getByLabel("Current password").fill("wrong-one-123");
    await card.getByLabel("New password", { exact: true }).fill("Turbine-Harbor-9051");
    await card.getByLabel("Confirm new password").fill("Turbine-Harbor-9051");
    await card.getByRole("button", { name: "Change password" }).click();
    await expect(page.getByTestId("password-error")).toHaveText("Current password is incorrect");

    await card.getByLabel("Current password").fill("Old-Passphrase-77");
    await card.getByRole("button", { name: "Change password" }).click();
    await expect(page.getByText(/Password changed/)).toBeVisible();
    await expect(card.getByLabel("Current password")).toHaveValue("");
    expect(bodies.at(-1)).toEqual({
      current_password: "Old-Passphrase-77",
      new_password: "Turbine-Harbor-9051",
    });
  });

  test("mismatched confirmation is caught before calling the API", async ({ page }) => {
    await page.route(api("/api/v1/auth/api-keys"), (route) => json(route, []));
    let called = false;
    await page.route("**/api/auth/password", (route) => {
      called = true;
      return json(route, { ok: true });
    });
    await page.goto("/settings/account");
    const card = page.getByTestId("change-password");
    await card.getByLabel("Current password").fill("Old-Passphrase-77");
    await card.getByLabel("New password", { exact: true }).fill("Turbine-Harbor-9051");
    await card.getByLabel("Confirm new password").fill("Turbine-Harbor-905");
    await expect(card.getByText("Does not match")).toBeVisible();
    await expect(card.getByRole("button", { name: "Change password" })).toBeDisabled();
    expect(called).toBe(false);
  });

  test("log out everywhere revokes sessions and returns to the login page", async ({ page }) => {
    await page.route(api("/api/v1/auth/api-keys"), (route) => json(route, []));
    let revoked = false;
    await page.route("**/api/auth/logout-all", (route) => {
      revoked = true;
      return json(route, { ok: true });
    });
    await page.goto("/settings/account");
    const card = page.getByTestId("sessions-card");
    await card.getByRole("button", { name: "Log out everywhere" }).click();
    await card.getByRole("button", { name: "Confirm: log out everywhere" }).click();
    await expect(page).toHaveURL(/\/login/);
    expect(revoked).toBe(true);
  });
});

test("customers can change their password from the portal", async ({ page, context }) => {
  await signInAs(context, "customer");
  await page.route(api("/api/v1/customer/me"), (route) =>
    json(route, { id: "c-1", name: "Demo Member", address: "", tariff_id: null }),
  );
  await page.goto("/portal/account");
  await expect(page.getByTestId("portal-account")).toBeVisible();
  await expect(page.getByTestId("change-password")).toBeVisible();
  await expect(page.getByTestId("my-api-keys")).toHaveCount(0);
});

test("login shows the lockout message on 429", async ({ page }) => {
  await page.route("**/api/auth/login", (route) =>
    json(route, { detail: "Too many failed attempts. Wait a few minutes and try again." }, 429),
  );
  await page.goto("/login");
  await page.getByLabel("Username").fill("admin");
  await page.getByLabel("Password").fill("whatever-123");
  await page.getByRole("button", { name: /sign in/i }).click();
  await expect(page.getByTestId("login-error")).toHaveText(/Too many failed attempts/);
});
