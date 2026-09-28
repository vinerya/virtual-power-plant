"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { UserPlus } from "lucide-react";
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
  createUser,
  deleteUser,
  listApiKeys,
  listUsers,
  resetUserPassword,
  revokeUserSessions,
  updateUser,
  type User,
  type UserRole,
} from "@/lib/api/users";
import { ApiKeyTable, PASSWORD_MIN_LENGTH, formatWhen, useRevokeApiKey } from "./account-view";

const USERS_KEY = ["users"] as const;
const ALL_KEYS_KEY = ["api-keys", "all"] as const;

function CreateUserCard() {
  const qc = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("viewer");
  const [error, setError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: () => createUser({ username: username.trim(), password, role }),
    onSuccess: (u) => {
      toast.success(`User ${u.username} created`);
      setUsername("");
      setPassword("");
      setError(null);
      qc.invalidateQueries({ queryKey: USERS_KEY });
    },
    onError: (e) => setError(apiErrorMessage(e, "Could not create the user")),
  });
  return (
    <Card data-testid="create-user">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <UserPlus className="h-4 w-4" aria-hidden="true" />
          Add user
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            mutation.mutate();
          }}
        >
          <Field label="New username" className="min-w-[12rem] flex-1">
            <Input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="off"
              pattern="[A-Za-z0-9_-]{3,64}"
              title="3-64 letters, digits, '_' or '-'"
              required
            />
          </Field>
          <Field label="Initial password" className="min-w-[12rem] flex-1">
            <Input
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              minLength={PASSWORD_MIN_LENGTH}
              required
            />
          </Field>
          <Field label="Role" className="w-40">
            <Select value={role} onChange={(e) => setRole(e.target.value as UserRole)}>
              {USER_ROLES.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </Select>
          </Field>
          <Button type="submit" disabled={mutation.isPending}>
            Create user
          </Button>
        </form>
        {error && (
          <p role="alert" className="mt-2 text-sm text-destructive" data-testid="create-user-error">
            {error}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function ResetPasswordForm({ user, onDone }: { user: User; onDone: () => void }) {
  const [password, setPassword] = useState("");
  const mutation = useMutation({
    mutationFn: () => resetUserPassword(user.id, password),
    onSuccess: () => {
      toast.success(`Password of ${user.username} reset; their sessions were signed out`);
      onDone();
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Password reset failed")),
  });
  return (
    <form
      className="flex flex-wrap items-end gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        mutation.mutate();
      }}
    >
      <Field label={`New password for ${user.username}`} className="min-w-[14rem] flex-1">
        <Input
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </Field>
      <Button type="submit" size="sm" disabled={!password || mutation.isPending}>
        Set password
      </Button>
      <Button type="button" size="sm" variant="ghost" onClick={onDone}>
        Cancel
      </Button>
    </form>
  );
}

function UserRow({ user, isSelf }: { user: User; isSelf: boolean }) {
  const qc = useQueryClient();
  const [resetting, setResetting] = useState(false);
  const [confirmDeactivate, setConfirmDeactivate] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const update = useMutation({
    mutationFn: (patch: { role?: UserRole; is_active?: boolean }) => updateUser(user.id, patch),
    onSuccess: (u, patch) => {
      if (patch.role) toast.success(`${u.username} is now ${u.role}; their sessions were signed out`);
      else if (patch.is_active === false)
        toast.success(`${u.username} deactivated; sessions and API keys revoked`);
      else toast.success(`${u.username} activated`);
      setConfirmDeactivate(false);
      qc.invalidateQueries({ queryKey: USERS_KEY });
      qc.invalidateQueries({ queryKey: ["api-keys"] });
    },
    onError: (e) => {
      toast.error(apiErrorMessage(e, "Update failed"));
      qc.invalidateQueries({ queryKey: USERS_KEY });
    },
  });
  const remove = useMutation({
    mutationFn: () => deleteUser(user.id),
    onSuccess: () => {
      toast.success(`${user.username} deleted; their API keys were removed`);
      setConfirmDelete(false);
      qc.invalidateQueries({ queryKey: USERS_KEY });
      qc.invalidateQueries({ queryKey: ["api-keys"] });
    },
    onError: (e) => {
      toast.error(apiErrorMessage(e, "Could not delete the user"));
      setConfirmDelete(false);
    },
  });
  const revoke = useMutation({
    mutationFn: () => revokeUserSessions(user.id),
    onSuccess: () => toast.success(`All sessions of ${user.username} revoked`),
    onError: (e) => toast.error(apiErrorMessage(e, "Could not revoke sessions")),
  });

  return (
    <>
      <tr className="border-t align-middle" data-testid={`user-row-${user.username}`}>
        <td className="py-2 pr-3 font-medium">
          {user.username}
          {isSelf && (
            <Badge variant="secondary" className="ml-2">
              you
            </Badge>
          )}
        </td>
        <td className="py-2 pr-3">
          <Select
            aria-label={`Role of ${user.username}`}
            value={user.role}
            disabled={isSelf || update.isPending}
            title={isSelf ? "Another admin must change your role" : undefined}
            onChange={(e) => update.mutate({ role: e.target.value as UserRole })}
            className="h-8 min-w-[8rem]"
          >
            {USER_ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </Select>
        </td>
        <td className="py-2 pr-3">
          {user.is_active ? (
            <Badge variant="success">active</Badge>
          ) : (
            <Badge variant="destructive">inactive</Badge>
          )}
        </td>
        <td className="py-2 pr-3 text-muted-foreground">{formatWhen(user.last_login_at)}</td>
        <td className="py-2 pr-3 tabular-nums">{user.api_key_count}</td>
        <td className="py-2 text-right">
          <span className="inline-flex flex-wrap justify-end gap-1">
            {user.is_active ? (
              confirmDeactivate ? (
                <>
                  <Button
                    size="sm"
                    variant="destructive"
                    disabled={update.isPending}
                    onClick={() => update.mutate({ is_active: false })}
                  >
                    Confirm deactivate
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setConfirmDeactivate(false)}>
                    Cancel
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={isSelf}
                  aria-label={`Deactivate ${user.username}`}
                  onClick={() => setConfirmDeactivate(true)}
                >
                  Deactivate
                </Button>
              )
            ) : (
              <Button
                size="sm"
                variant="outline"
                aria-label={`Activate ${user.username}`}
                disabled={update.isPending}
                onClick={() => update.mutate({ is_active: true })}
              >
                Activate
              </Button>
            )}
            {!isSelf && (
              <Button
                size="sm"
                variant="outline"
                aria-label={`Reset password of ${user.username}`}
                onClick={() => setResetting((v) => !v)}
              >
                Reset password
              </Button>
            )}
            <Button
              size="sm"
              variant="outline"
              aria-label={`Sign out ${user.username} everywhere`}
              disabled={revoke.isPending}
              onClick={() => revoke.mutate()}
            >
              Revoke sessions
            </Button>
            {!isSelf &&
              (confirmDelete ? (
                <>
                  <Button
                    size="sm"
                    variant="destructive"
                    disabled={remove.isPending}
                    onClick={() => remove.mutate()}
                  >
                    Confirm delete
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(false)}>
                    Cancel
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  aria-label={`Delete ${user.username}`}
                  onClick={() => setConfirmDelete(true)}
                >
                  Delete
                </Button>
              ))}
          </span>
        </td>
      </tr>
      {resetting && (
        <tr>
          <td colSpan={6} className="pb-3">
            <ResetPasswordForm user={user} onDone={() => setResetting(false)} />
          </td>
        </tr>
      )}
    </>
  );
}

function UsersTable() {
  const session = useSession();
  const users = useQuery({ queryKey: USERS_KEY, queryFn: listUsers });
  if (users.isLoading) return <Skeleton className="h-32 w-full" />;
  if (users.isError) {
    return <ErrorState title="Could not load users" error={users.error} onRetry={users.refetch} />;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="users-table">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="py-2 pr-3 font-medium">User</th>
            <th className="py-2 pr-3 font-medium">Role</th>
            <th className="py-2 pr-3 font-medium">Status</th>
            <th className="py-2 pr-3 font-medium">Last login</th>
            <th className="py-2 pr-3 font-medium">API keys</th>
            <th className="py-2 font-medium">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {(users.data ?? []).map((u) => (
            <UserRow key={u.id} user={u} isSelf={u.username === session.data?.username} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AllApiKeysCard() {
  const keys = useQuery({ queryKey: ALL_KEYS_KEY, queryFn: () => listApiKeys(true) });
  const revoke = useRevokeApiKey([["api-keys"], USERS_KEY]);
  return (
    <Card data-testid="all-api-keys">
      <CardHeader>
        <CardTitle className="text-base">All API keys</CardTitle>
      </CardHeader>
      <CardContent>
        {keys.isLoading ? (
          <Skeleton className="h-16 w-full" />
        ) : keys.isError ? (
          <ErrorState title="Could not load API keys" error={keys.error} onRetry={keys.refetch} />
        ) : (
          <ApiKeyTable
            keys={keys.data ?? []}
            showOwner
            onRevoke={(k) => revoke.mutate(k)}
            revokingId={revoke.isPending ? revoke.variables?.id : null}
          />
        )}
      </CardContent>
    </Card>
  );
}

/** Settings → Users & API keys (admin). The API enforces the admin role. */
export function UsersView() {
  const session = useSession();
  if (session.isLoading) return <Skeleton className="h-32 w-full" />;
  if (session.data?.role !== "admin") {
    return (
      <p className="text-sm text-muted-foreground" data-testid="users-admin-only">
        Only admins can manage users and API keys. Your own password and keys are under Account.
      </p>
    );
  }
  return (
    <div className="space-y-4" data-testid="users-view">
      <CreateUserCard />
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Users</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-xs text-muted-foreground">
            Changing a role or deactivating a user signs them out everywhere; deactivation also
            revokes their API keys. Deleting a user removes the account and its API keys for good
            (the audit log keeps their name). You cannot demote, deactivate or delete yourself or
            the last active admin.
          </p>
          <UsersTable />
        </CardContent>
      </Card>
      <AllApiKeysCard />
    </div>
  );
}
