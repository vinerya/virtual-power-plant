// Demo/mock mode.
//
// Mock data is strictly opt-in: set NEXT_PUBLIC_USE_MOCKS=1 (e.g. for a
// backend-less demo). Without it, a missing or failing backend endpoint
// surfaces as an error in the UI instead of silently rendering fake data.
//
// NEXT_PUBLIC_* values are inlined at build time, so the flag must be set
// when running `next build` / `next dev`, not only at `next start`.

export const USE_MOCKS =
  process.env.NEXT_PUBLIC_USE_MOCKS === "1" ||
  process.env.NEXT_PUBLIC_USE_MOCKS === "true";

/**
 * Call `fn`; in mock mode, fall back to `mock()` when the request fails for
 * any reason (404, 5xx, backend unreachable). Outside mock mode errors
 * propagate unchanged so the UI can render a proper error state.
 */
export async function withMockFallback<T>(
  fn: () => Promise<T>,
  mock: () => T,
): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    if (USE_MOCKS) {
      if (process.env.NODE_ENV !== "production") {
        console.warn("[mock mode] using mock data after API error:", e);
      }
      return mock();
    }
    throw e;
  }
}
