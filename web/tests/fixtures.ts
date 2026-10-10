import { test as base, expect } from "@playwright/test";

export * from "@playwright/test";

// Every spec imports `test` from here, so a page the Content-Security-Policy
// in middleware.ts breaks fails the test that opened it. Chromium logs most
// violations to the console ("Refused to load the script … because it
// violates the following Content Security Policy directive …"); the
// `securitypolicyviolation` listener also reports the ones it does not, such
// as a blocked eval() whose EvalError the app catches.
const CSP_VIOLATION =
  /Content Security Policy|CSP violation|Refused to (load|execute|connect|apply|frame|create|evaluate)/i;

export const test = base.extend<{ cspViolations: string[] }>({
  cspViolations: [
    async ({ context }, use) => {
      const violations: string[] = [];
      await context.addInitScript(() => {
        document.addEventListener("securitypolicyviolation", (e) => {
          console.error(
            `CSP violation: ${e.effectiveDirective} blocked ${e.blockedURI || "(inline)"}`,
          );
        });
      });
      context.on("console", (msg) => {
        if (CSP_VIOLATION.test(msg.text())) violations.push(msg.text());
      });
      await use(violations);
      expect(violations, "Content-Security-Policy violations").toEqual([]);
    },
    { auto: true },
  ],
});
