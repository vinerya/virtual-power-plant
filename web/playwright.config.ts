import { defineConfig, devices } from "@playwright/test";

// The specs stub every backend call with `page.route()`, so no FastAPI
// server is needed. The app is NOT built in mock mode: the tests exercise
// the real API client code paths against the stubbed responses.
//
// Env:
//   E2E_BASE_URL       — test an already-running app instead of starting one
//   E2E_NO_SERVER=1    — don't start a web server
//   E2E_PORT           — port for the dev server (default 3000)
//   PLAYWRIGHT_CHROMIUM_EXECUTABLE — use a preinstalled Chromium whose
//                        revision differs from the one this Playwright
//                        version expects (skips `playwright install`).

const PORT = Number(process.env.E2E_PORT || 3000);
const BASE_URL = process.env.E2E_BASE_URL || `http://localhost:${PORT}`;
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined;

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : [["list"]],
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: executablePath ? { executablePath } : {},
      },
    },
  ],
  webServer:
    process.env.E2E_NO_SERVER || process.env.E2E_BASE_URL
      ? undefined
      : {
          // CI runs against a production build (`pnpm build` first) for
          // speed and fidelity; locally `pnpm dev` is used.
          command: process.env.CI
            ? `pnpm start --port ${PORT}`
            : `pnpm dev --port ${PORT}`,
          url: `${BASE_URL}/login`,
          reuseExistingServer: !process.env.CI,
          timeout: 180_000,
        },
});
