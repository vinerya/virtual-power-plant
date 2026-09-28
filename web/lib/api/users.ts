// Users, passwords and API keys (/api/v1/users, /api/v1/auth/*).
//
// Password change and "log out everywhere" go through dedicated Next.js
// routes (/api/auth/password, /api/auth/logout-all) rather than the generic
// proxy: both revoke the session cookie's token, so the route handler has
// to replace (or clear) the httpOnly cookie in the same response.
import { z } from "zod";
import { api, type ApiError } from "./client";
import { parseResponse } from "./errors";

export const USER_ROLES = ["admin", "operator", "viewer", "researcher", "customer"] as const;
export type UserRole = (typeof USER_ROLES)[number];

const role = z.enum(USER_ROLES);

export const userSchema = z.object({
  id: z.string(),
  username: z.string(),
  role,
  is_active: z.boolean(),
  created_at: z.string(),
  last_login_at: z.string().nullable().optional(),
  api_key_count: z.number().optional().default(0),
});
export type User = z.infer<typeof userSchema>;

export const apiKeySchema = z.object({
  id: z.string(),
  name: z.string(),
  role,
  is_active: z.boolean(),
  created_at: z.string(),
  last_used_at: z.string().nullable().optional(),
  key_prefix: z.string().nullable().optional(),
  user_id: z.string(),
  username: z.string().nullable().optional(),
});
export type ApiKey = z.infer<typeof apiKeySchema>;

const createdKeySchema = z.object({
  id: z.string(),
  name: z.string(),
  key: z.string(),
  role,
  created_at: z.string(),
  key_prefix: z.string().nullable().optional(),
});
export type CreatedApiKey = z.infer<typeof createdKeySchema>;

// -- admin -------------------------------------------------------------------

export async function listUsers(): Promise<User[]> {
  return parseResponse(z.array(userSchema), await api.get("/api/v1/users"), "users");
}

export async function createUser(body: {
  username: string;
  password: string;
  role: UserRole;
}): Promise<User> {
  return parseResponse(userSchema, await api.post("/api/v1/users", body), "user");
}

export async function updateUser(
  id: string,
  patch: { role?: UserRole; is_active?: boolean },
): Promise<User> {
  return parseResponse(
    userSchema,
    await api.patch(`/api/v1/users/${encodeURIComponent(id)}`, patch),
    "user",
  );
}

export function resetUserPassword(id: string, newPassword: string): Promise<void> {
  return api.post(`/api/v1/users/${encodeURIComponent(id)}/password`, {
    new_password: newPassword,
  });
}

export function revokeUserSessions(id: string): Promise<void> {
  return api.post(`/api/v1/users/${encodeURIComponent(id)}/revoke-sessions`);
}

// -- API keys ----------------------------------------------------------------

export async function listApiKeys(all = false): Promise<ApiKey[]> {
  return parseResponse(
    z.array(apiKeySchema),
    await api.get(`/api/v1/auth/api-keys${all ? "?all=true" : ""}`),
    "API keys",
  );
}

export async function createApiKey(body: { name: string; role: UserRole }): Promise<CreatedApiKey> {
  return parseResponse(createdKeySchema, await api.post("/api/v1/auth/api-keys", body), "API key");
}

export function revokeApiKey(id: string): Promise<void> {
  return api.delete(`/api/v1/auth/api-keys/${encodeURIComponent(id)}`);
}

// -- self-service (Next.js routes that manage the session cookie) ----------------

async function postSession(path: string, body?: unknown): Promise<void> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => undefined);
    const err = new Error(`${path} ${res.status}`) as ApiError;
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
}

/** Change the signed-in user's password; the session cookie is replaced. */
export function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  return postSession("/api/auth/password", {
    current_password: currentPassword,
    new_password: newPassword,
  });
}

/** Revoke every session of the signed-in user (this one included). */
export function logoutEverywhere(): Promise<void> {
  return postSession("/api/auth/logout-all");
}
