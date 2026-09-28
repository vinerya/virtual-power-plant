"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Copy, KeyRound, LogOut } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { apiErrorMessage } from "@/lib/api/errors";
import { useSession } from "@/lib/api/session";
import {
  USER_ROLES,
  changePassword,
  createApiKey,
  listApiKeys,
  logoutEverywhere,
  revokeApiKey,
  type ApiKey,
  type CreatedApiKey,
  type UserRole,
} from "@/lib/api/users";

/** Mirrors the backend default (VPP_PASSWORD_MIN_LENGTH); the API decides. */
export const PASSWORD_MIN_LENGTH = 12;

export function formatWhen(ts: string | null | undefined): string {
  if (!ts) return "never";
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleString();
}

export function ChangePasswordCard() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => changePassword(current, next),
    onSuccess: () => {
      setCurrent("");
      setNext("");
      setConfirm("");
      setError(null);
      toast.success("Password changed. Your other sessions were signed out.");
    },
    onError: (e) => setError(apiErrorMessage(e, "Password change failed")),
  });

  const mismatch = confirm.length > 0 && next !== confirm;

  return (
    <Card data-testid="change-password">
      <CardHeader>
        <CardTitle className="text-base">Change password</CardTitle>
      </CardHeader>
      <CardContent>
        <form
          className="grid max-w-md gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            if (next !== confirm) {
              setError("The new passwords do not match");
              return;
            }
            mutation.mutate();
          }}
        >
          <Field label="Current password">
            <Input
              type="password"
              autoComplete="current-password"
              value={current}
              onChange={(e) => setCurrent(e.target.value)}
              required
            />
          </Field>
          <Field
            label="New password"
            hint={`At least ${PASSWORD_MIN_LENGTH} characters; common passwords and your username are refused.`}
          >
            <Input
              type="password"
              autoComplete="new-password"
              value={next}
              onChange={(e) => setNext(e.target.value)}
              required
            />
          </Field>
          <Field label="Confirm new password" error={mismatch ? "Does not match" : null}>
            <Input
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              required
            />
          </Field>
          {error && (
            <p role="alert" className="text-sm text-destructive" data-testid="password-error">
              {error}
            </p>
          )}
          <div>
            <Button type="submit" disabled={mutation.isPending || !current || !next || mismatch}>
              {mutation.isPending ? "Saving…" : "Change password"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

export function SessionsCard() {
  const [confirming, setConfirming] = useState(false);
  const mutation = useMutation({
    mutationFn: logoutEverywhere,
    onSuccess: () => {
      window.location.href = "/login";
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Could not sign out everywhere")),
  });
  return (
    <Card data-testid="sessions-card">
      <CardHeader>
        <CardTitle className="text-base">Sessions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-muted-foreground">
          Sign out of every browser and device, including this one. API keys keep working;
          revoke them separately.
        </p>
        {confirming ? (
          <div className="flex gap-2">
            <Button
              variant="destructive"
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending}
            >
              <LogOut className="h-4 w-4" aria-hidden="true" />
              Confirm: log out everywhere
            </Button>
            <Button variant="outline" onClick={() => setConfirming(false)}>
              Cancel
            </Button>
          </div>
        ) : (
          <Button variant="outline" onClick={() => setConfirming(true)}>
            <LogOut className="h-4 w-4" aria-hidden="true" />
            Log out everywhere
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

/** Shows a freshly created key exactly once. */
export function NewKeyPanel({ created, onDone }: { created: CreatedApiKey; onDone: () => void }) {
  return (
    <div
      role="status"
      data-testid="new-api-key"
      className="space-y-2 rounded-md border border-amber-500/50 bg-amber-500/5 p-3 text-sm"
    >
      <p className="font-medium">
        API key “{created.name}” created. Copy it now — it will not be shown again.
      </p>
      <div className="flex items-center gap-2">
        <code
          className="flex-1 break-all rounded bg-background px-2 py-1 font-mono text-xs"
          data-testid="new-api-key-value"
        >
          {created.key}
        </code>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => {
            navigator.clipboard
              ?.writeText(created.key)
              .then(() => toast.success("Copied"))
              .catch(() => toast.error("Copy failed; select the key and copy it manually"));
          }}
        >
          <Copy className="h-4 w-4" aria-hidden="true" />
          Copy
        </Button>
      </div>
      <Button type="button" size="sm" variant="secondary" onClick={onDone}>
        I have stored it
      </Button>
    </div>
  );
}

export function ApiKeyTable({
  keys,
  showOwner,
  onRevoke,
  revokingId,
}: {
  keys: ApiKey[];
  showOwner?: boolean;
  onRevoke: (key: ApiKey) => void;
  revokingId?: string | null;
}) {
  const [confirmId, setConfirmId] = useState<string | null>(null);
  if (!keys.length) {
    return <p className="text-sm text-muted-foreground">No active API keys.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="api-key-table">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="py-2 pr-3 font-medium">Name</th>
            {showOwner && <th className="py-2 pr-3 font-medium">Owner</th>}
            <th className="py-2 pr-3 font-medium">Role</th>
            <th className="py-2 pr-3 font-medium">Key</th>
            <th className="py-2 pr-3 font-medium">Created</th>
            <th className="py-2 pr-3 font-medium">Last used</th>
            <th className="py-2 font-medium">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k.id} className="border-t" data-testid={`api-key-${k.id}`}>
              <td className="py-2 pr-3 font-medium">{k.name}</td>
              {showOwner && <td className="py-2 pr-3">{k.username ?? k.user_id}</td>}
              <td className="py-2 pr-3">
                <Badge variant="outline">{k.role}</Badge>
              </td>
              <td className="py-2 pr-3 font-mono text-xs">
                {k.key_prefix ? `${k.key_prefix}…` : "—"}
              </td>
              <td className="py-2 pr-3 text-muted-foreground">{formatWhen(k.created_at)}</td>
              <td className="py-2 pr-3 text-muted-foreground">{formatWhen(k.last_used_at)}</td>
              <td className="py-2 text-right">
                {confirmId === k.id ? (
                  <span className="inline-flex gap-1">
                    <Button
                      size="sm"
                      variant="destructive"
                      disabled={revokingId === k.id}
                      onClick={() => onRevoke(k)}
                    >
                      Confirm revoke
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setConfirmId(null)}>
                      Cancel
                    </Button>
                  </span>
                ) : (
                  <Button
                    size="sm"
                    variant="outline"
                    aria-label={`Revoke API key ${k.name}`}
                    onClick={() => setConfirmId(k.id)}
                  >
                    Revoke
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function useRevokeApiKey(queryKeys: readonly (readonly unknown[])[]) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (key: ApiKey) => revokeApiKey(key.id),
    onSuccess: (_d, key) => {
      toast.success(`API key “${key.name}” revoked`);
      for (const k of queryKeys) qc.invalidateQueries({ queryKey: k });
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Could not revoke the key")),
  });
}

function MyApiKeysCard({ role }: { role: UserRole }) {
  const qc = useQueryClient();
  const keys = useQuery({ queryKey: ["api-keys", "mine"], queryFn: () => listApiKeys(false) });
  const [name, setName] = useState("");
  const [keyRole, setKeyRole] = useState<UserRole>(role);
  const [created, setCreated] = useState<CreatedApiKey | null>(null);
  const create = useMutation({
    mutationFn: () => createApiKey({ name: name.trim(), role: keyRole }),
    onSuccess: (k) => {
      setCreated(k);
      setName("");
      qc.invalidateQueries({ queryKey: ["api-keys"] });
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Could not create the key")),
  });
  const revoke = useRevokeApiKey([["api-keys"]]);
  // Non-admins can only mint keys with their own role (backend rule).
  const roleChoices = role === "admin" ? USER_ROLES.filter((r) => r !== "customer") : [role];

  return (
    <Card data-testid="my-api-keys">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <KeyRound className="h-4 w-4" aria-hidden="true" />
          My API keys
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) create.mutate();
          }}
        >
          <Field label="Key name" className="min-w-[14rem] flex-1">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. scada-bridge"
              maxLength={128}
            />
          </Field>
          <Field label="Key role" className="w-40">
            <Select value={keyRole} onChange={(e) => setKeyRole(e.target.value as UserRole)}>
              {roleChoices.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </Select>
          </Field>
          <Button type="submit" disabled={!name.trim() || create.isPending}>
            Create key
          </Button>
        </form>
        {created && <NewKeyPanel created={created} onDone={() => setCreated(null)} />}
        {keys.isLoading ? (
          <Skeleton className="h-16 w-full" />
        ) : keys.isError ? (
          <ErrorState title="Could not load API keys" error={keys.error} onRetry={keys.refetch} />
        ) : (
          <ApiKeyTable
            keys={keys.data ?? []}
            onRevoke={(k) => revoke.mutate(k)}
            revokingId={revoke.isPending ? revoke.variables?.id : null}
          />
        )}
      </CardContent>
    </Card>
  );
}

/** Settings → Account: password, sessions and (operator-side) own API keys. */
export function AccountView() {
  const session = useSession();
  const role = session.data?.role;
  return (
    <div className="space-y-4" data-testid="account-view">
      {session.data?.username && (
        <p className="text-sm text-muted-foreground">
          Signed in as <span className="font-medium text-foreground">{session.data.username}</span>
          {role ? ` (${role})` : ""}
        </p>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        <ChangePasswordCard />
        <SessionsCard />
      </div>
      {role && role !== "customer" && <MyApiKeysCard role={role} />}
    </div>
  );
}
