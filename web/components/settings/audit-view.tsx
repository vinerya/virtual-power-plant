"use client";

import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/lib/api/session";
import { listAudit, type AuditEntry } from "@/lib/api/users";
import { formatWhen } from "./account-view";

const PAGE_SIZE = 50;

const ACTION_GROUPS = [
  { value: "", label: "All actions" },
  { value: "auth.", label: "Sign-in & sessions" },
  { value: "api_key.", label: "API keys" },
  { value: "user.", label: "User administration" },
  { value: "control.", label: "Device & DR control" },
  { value: "market.", label: "Market orders" },
  { value: "config.", label: "Configuration" },
] as const;

function OutcomeBadge({ outcome }: { outcome: string }) {
  if (outcome === "success") return <Badge variant="success">success</Badge>;
  if (outcome === "denied") return <Badge variant="secondary">denied</Badge>;
  return <Badge variant="destructive">{outcome}</Badge>;
}

function target(e: AuditEntry): string {
  if (!e.target_type && !e.target_id) return "";
  const name = typeof e.details?.username === "string" ? ` (${e.details.username})` : "";
  return `${e.target_type ?? ""} ${e.target_id ?? ""}${name}`.trim();
}

function detailsText(e: AuditEntry): string {
  const d = e.details ?? {};
  return Object.keys(d).length ? JSON.stringify(d) : "";
}

/** Settings → Audit log (admin). The API enforces the admin role. */
export function AuditView() {
  const session = useSession();
  const [actor, setActor] = useState("");
  const [action, setAction] = useState("");
  const [outcome, setOutcome] = useState("");
  const [offset, setOffset] = useState(0);
  const isAdmin = session.data?.role === "admin";

  const query = useQuery({
    queryKey: ["audit", actor, action, outcome, offset],
    queryFn: () =>
      listAudit({ actor: actor.trim(), action, outcome, limit: PAGE_SIZE, offset }),
    enabled: isAdmin,
    placeholderData: keepPreviousData,
  });

  if (session.isLoading) return <Skeleton className="h-32 w-full" />;
  if (!isAdmin) {
    return (
      <p className="text-sm text-muted-foreground" data-testid="audit-admin-only">
        Only admins can read the audit log.
      </p>
    );
  }

  const total = query.data?.total ?? null;
  const entries = query.data?.entries ?? [];
  const from = entries.length ? offset + 1 : 0;
  const to = offset + entries.length;
  const hasNext = total !== null ? to < total : entries.length === PAGE_SIZE;

  return (
    <Card data-testid="audit-view">
      <CardHeader>
        <CardTitle className="text-base">Audit log</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Sign-ins, session and API-key changes, user administration and privileged control
          actions, newest first. Passwords, tokens and keys are never recorded.
        </p>
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            setOffset(0);
            query.refetch();
          }}
        >
          <Field label="Actor (username or id)" className="min-w-[12rem] flex-1">
            <Input
              value={actor}
              onChange={(e) => {
                setActor(e.target.value);
                setOffset(0);
              }}
              autoComplete="off"
            />
          </Field>
          <Field label="Action" className="w-56">
            <Select
              value={action}
              onChange={(e) => {
                setAction(e.target.value);
                setOffset(0);
              }}
            >
              {ACTION_GROUPS.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Outcome" className="w-40">
            <Select
              value={outcome}
              onChange={(e) => {
                setOutcome(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">Any</option>
              <option value="success">success</option>
              <option value="failure">failure</option>
              <option value="denied">denied</option>
            </Select>
          </Field>
        </form>

        {query.isLoading ? (
          <Skeleton className="h-32 w-full" />
        ) : query.isError ? (
          <ErrorState
            title="Could not load the audit log"
            error={query.error}
            onRetry={query.refetch}
          />
        ) : entries.length === 0 ? (
          <p className="text-sm text-muted-foreground" data-testid="audit-empty">
            No matching entries.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="audit-table">
              <thead className="text-left text-xs text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3 font-medium">When</th>
                  <th className="py-2 pr-3 font-medium">Actor</th>
                  <th className="py-2 pr-3 font-medium">Action</th>
                  <th className="py-2 pr-3 font-medium">Outcome</th>
                  <th className="py-2 pr-3 font-medium">Target</th>
                  <th className="py-2 pr-3 font-medium">Client IP</th>
                  <th className="py-2 font-medium">Details</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e) => (
                  <tr key={e.id} className="border-t align-top" data-testid={`audit-row-${e.id}`}>
                    <td className="py-2 pr-3 whitespace-nowrap text-muted-foreground">
                      {formatWhen(e.ts)}
                    </td>
                    <td className="py-2 pr-3">{e.actor_username ?? "anonymous"}</td>
                    <td className="py-2 pr-3 font-mono text-xs">{e.action}</td>
                    <td className="py-2 pr-3">
                      <OutcomeBadge outcome={e.outcome} />
                    </td>
                    <td className="py-2 pr-3 break-all text-xs">{target(e)}</td>
                    <td className="py-2 pr-3 font-mono text-xs">{e.client_ip ?? ""}</td>
                    <td className="py-2 break-all font-mono text-xs text-muted-foreground">
                      {detailsText(e)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="flex items-center justify-between gap-2 text-sm">
          <span className="text-muted-foreground" data-testid="audit-range">
            {from}-{to}
            {total !== null ? ` of ${total}` : ""}
          </span>
          <span className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={offset === 0 || query.isFetching}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              Previous
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={!hasNext || query.isFetching}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Next
            </Button>
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
