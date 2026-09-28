import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import { test, expect } from "@playwright/test";

/**
 * The server-side proxy (/api/proxy/*) and auth routes forward the client
 * address (X-Forwarded-For / X-Real-IP) to FastAPI, so the API's per-IP rate
 * limiter can tell console users apart. A fake backend on E2E_BACKEND_PORT
 * (the web server's API_BASE_URL, see playwright.config.ts) records the
 * headers it receives.
 */

const PORT = Number(process.env.E2E_PORT || 3000);
const BACKEND_PORT = Number(process.env.E2E_BACKEND_PORT || PORT + 1);

test.describe.configure({ mode: "serial" });
test.skip(
  !!process.env.E2E_BASE_URL,
  "needs the web server started by Playwright (API_BASE_URL -> fake backend)",
);

// /api/proxy/* requires a session cookie (middleware); its value is opaque here.
const SESSION = { cookie: "vpp_session=fake.jwt.token" };

let server: Server;
const seen: { path: string; headers: IncomingHttpHeaders }[] = [];

test.beforeAll(async () => {
  server = createServer((req, res) => {
    seen.push({ path: req.url || "", headers: req.headers });
    res.setHeader("content-type", "application/json");
    if (req.url === "/api/v1/auth/token") {
      res.end(JSON.stringify({ access_token: "t", token_type: "bearer", expires_in: 60 }));
      return;
    }
    res.end(JSON.stringify({ ok: true }));
  });
  await new Promise<void>((resolve) => server.listen(BACKEND_PORT, "127.0.0.1", resolve));
});

test.afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

test.beforeEach(() => {
  seen.length = 0;
});

test("proxy forwards the client address chain", async ({ request }) => {
  const res = await request.get("/api/proxy/api/v1/resources", {
    headers: { ...SESSION, "x-forwarded-for": "198.51.100.7" },
  });
  expect(res.status()).toBe(200);
  const hit = seen.find((s) => s.path === "/api/v1/resources");
  expect(hit).toBeDefined();
  expect(hit!.headers["x-forwarded-for"]).toBe("198.51.100.7");
  expect(hit!.headers["x-real-ip"]).toBe("198.51.100.7");
});

test("proxy fills in the socket address when none is supplied", async ({ request }) => {
  const res = await request.get("/api/proxy/api/v1/resources", { headers: SESSION });
  expect(res.status()).toBe(200);
  const hit = seen.find((s) => s.path === "/api/v1/resources");
  const xff = String(hit!.headers["x-forwarded-for"] || "");
  // Next.js sets X-Forwarded-For from the connection (loopback here).
  expect(xff).toMatch(/127\.0\.0\.1|::1/);
  expect(hit!.headers["x-real-ip"]).toBe(xff.split(",").pop()!.trim());
});

test("login forwards the client address", async ({ request }) => {
  const res = await request.post("/api/auth/login", {
    data: { username: "u", password: "p" },
    headers: { "x-forwarded-for": "203.0.113.4, 198.51.100.9" },
  });
  expect(res.status()).toBe(200);
  const hit = seen.find((s) => s.path === "/api/v1/auth/token");
  expect(hit!.headers["x-forwarded-for"]).toBe("203.0.113.4, 198.51.100.9");
  expect(hit!.headers["x-real-ip"]).toBe("198.51.100.9");
});
