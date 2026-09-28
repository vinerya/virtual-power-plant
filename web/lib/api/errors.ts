// Helpers for turning FastAPI error payloads into something a human can read.
//
// FastAPI puts the useful information in `detail`, which can be:
//   - a string                     → HTTPException(detail="...")
//   - {message, ...}               → structured domain errors (trading, config)
//   - [{loc, msg, type}, ...]      → request-validation errors (422)
import { z } from "zod";
import type { ApiError } from "./client";

export function apiStatus(err: unknown): number | undefined {
  const s = (err as { status?: unknown } | null)?.status;
  return typeof s === "number" ? s : undefined;
}

/** `detail` from a failed request (already JSON-parsed by the client). */
export function apiDetail(err: unknown): unknown {
  const d = (err as ApiError | null)?.detail;
  if (d && typeof d === "object" && "detail" in (d as object)) {
    return (d as { detail: unknown }).detail;
  }
  return d;
}

const validationItem = z.object({
  loc: z.array(z.union([z.string(), z.number()])).optional(),
  msg: z.string(),
});

/**
 * Best human-readable message for a failed request, or `fallback` when the
 * server did not say anything useful.
 */
export function apiErrorMessage(err: unknown, fallback = "Request failed"): string {
  const detail = apiDetail(err);
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object") {
    if (Array.isArray(detail)) {
      const items = detail
        .map((d) => validationItem.safeParse(d))
        .filter((r) => r.success)
        .map((r) => {
          const loc = (r.data.loc ?? []).filter((p) => p !== "body").join(".");
          return loc ? `${loc}: ${r.data.msg}` : r.data.msg;
        });
      if (items.length) return items.join("; ");
    }
    const msg = (detail as { message?: unknown }).message;
    if (typeof msg === "string" && msg.trim()) return msg;
  }
  if (err instanceof z.ZodError) {
    return "The server response did not match the expected format.";
  }
  return fallback;
}

/**
 * Parse a response with a zod schema. A mismatch throws an Error whose
 * message names the offending path, so contract drift between the backend
 * and the UI shows up as a clear error instead of `undefined` in a table.
 */
export function parseResponse<S extends z.ZodTypeAny>(
  schema: S,
  data: unknown,
  what: string,
): z.output<S> {
  const r = schema.safeParse(data);
  if (r.success) return r.data as z.output<S>;
  const first = r.error.issues[0];
  const path = first?.path.join(".") || "(root)";
  throw new Error(
    `Unexpected ${what} response from the API at ${path}: ${first?.message ?? "invalid"}`,
  );
}
