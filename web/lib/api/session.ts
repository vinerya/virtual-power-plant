"use client";

// Current console session (username + role) for role-aware UI.
//
// UI gating is a convenience only: the backend enforces every permission
// (e.g. POST /trading/orders requires operator or admin). When the role is
// unknown (request failed) the UI hides write actions rather than offering
// buttons that would 403.
import { useQuery } from "@tanstack/react-query";
import { z } from "zod";
import { USE_MOCKS } from "./mocks";

export const ROLES = ["admin", "operator", "viewer", "customer"] as const;
export type Role = (typeof ROLES)[number];

const sessionSchema = z.object({
  authenticated: z.boolean(),
  audience: z.string().optional(),
  username: z.string().optional(),
  role: z.enum(ROLES).optional().catch(undefined),
});

export type Session = z.infer<typeof sessionSchema>;

export async function getSession(): Promise<Session> {
  if (USE_MOCKS) {
    return { authenticated: true, audience: "operator", username: "demo", role: "operator" };
  }
  const res = await fetch("/api/auth/me", { cache: "no-store", credentials: "same-origin" });
  const json = await res.json().catch(() => ({ authenticated: false }));
  const parsed = sessionSchema.safeParse(json);
  return parsed.success ? parsed.data : { authenticated: false };
}

export function useSession() {
  return useQuery({
    queryKey: ["session"],
    queryFn: getSession,
    staleTime: 5 * 60_000,
    retry: false,
  });
}

/** Roles allowed to change the fleet / markets (mirrors backend require_role). */
export function canOperate(role: Role | undefined): boolean {
  return role === "admin" || role === "operator";
}

/** Convenience hook: `{ role, canOperate, isLoading }`. */
export function useRole() {
  const q = useSession();
  const role = q.data?.role;
  return { role, canOperate: canOperate(role), isLoading: q.isLoading };
}
