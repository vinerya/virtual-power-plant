import { test, expect } from "./fixtures";

// Security headers from next.config.ts (static) and middleware.ts (CSP with
// a per-request nonce). Every spec also fails on a CSP violation logged by
// the browser (tests/fixtures.ts), which is what shows the policy fits.

function directives(csp: string): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const part of csp.split(";")) {
    const [name, ...values] = part.trim().split(/\s+/);
    if (name) out[name] = values;
  }
  return out;
}

test("pages carry the security headers and a nonce-based CSP", async ({ page }) => {
  const response = await page.goto("/login");
  expect(response).not.toBeNull();
  const headers = response!.headers();
  expect(headers["x-content-type-options"]).toBe("nosniff");
  expect(headers["x-frame-options"]).toBe("DENY");
  expect(headers["referrer-policy"]).toBe("strict-origin-when-cross-origin");
  expect(headers["permissions-policy"]).toContain("camera=()");
  expect(headers["x-powered-by"]).toBeUndefined();

  const csp = directives(headers["content-security-policy"] ?? "");
  expect(csp["frame-ancestors"]).toEqual(["'none'"]);
  expect(csp["object-src"]).toEqual(["'none'"]);
  expect(csp["script-src"]).toContain("'strict-dynamic'");
  expect(csp["script-src"]).not.toContain("'unsafe-inline'");
  const nonce = csp["script-src"].find((s) => s.startsWith("'nonce-"))?.slice(7, -1);
  expect(nonce).toBeTruthy();
  if (!process.env.E2E_BASE_URL) {
    // The browser dials the API's WebSocket directly (WS_PUBLIC_URL, set
    // by playwright.config.ts for the server it starts).
    expect(csp["connect-src"]).toContain("ws://backend.test");
  }

  // Next.js stamped the nonce on the scripts it rendered...
  const html = await response!.text();
  expect(html).toContain(`nonce="${nonce}"`);
  // ...so the page hydrates and works.
  await expect(page.getByLabel("Username")).toBeVisible();
  await page.getByLabel("Username").fill("operator");
  await expect(page.getByLabel("Username")).toHaveValue("operator");

  // A fresh nonce on every request.
  const again = await page.request.get("/login");
  expect(again.headers()["content-security-policy"]).not.toContain(`'nonce-${nonce}'`);
});

test("route handler responses carry the static headers too", async ({ request }) => {
  const r = await request.get("/api/auth/me");
  expect(r.headers()["x-content-type-options"]).toBe("nosniff");
  expect(r.headers()["x-frame-options"]).toBe("DENY");
});
