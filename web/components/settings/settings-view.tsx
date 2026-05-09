"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { CheckCircle2, AlertCircle, Save, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { YamlEditor } from "./yaml-editor";
import { ConfigDiff } from "./config-diff";
import { applyConfig, getConfig, getConfigSchema } from "@/lib/api/config";
import type { ConfigValidationError } from "@/lib/api/types";

export function SettingsView() {
  const qc = useQueryClient();
  const live = useQuery({
    queryKey: ["config"],
    queryFn: getConfig,
    staleTime: 30_000,
  });
  const schema = useQuery({
    queryKey: ["config-schema"],
    queryFn: getConfigSchema,
    staleTime: 60 * 60 * 1000,
  });

  const [draft, setDraft] = useState<string>("");
  const [errors, setErrors] = useState<ConfigValidationError[]>([]);
  const [confirmOpen, setConfirmOpen] = useState(false);

  // Seed draft from live config the first time it loads.
  useEffect(() => {
    if (live.data && draft === "") setDraft(live.data.yaml);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live.data]);

  const dirty = useMemo(
    () => !!live.data && draft !== live.data.yaml,
    [live.data, draft],
  );

  async function runValidate() {
    setErrors([]);
    const errs: ConfigValidationError[] = [];
    let parsed: unknown = null;
    try {
      const yaml = await import("js-yaml");
      parsed = yaml.load(draft);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "YAML parse error";
      errs.push({ path: "$", message: msg });
    }
    if (errs.length === 0 && schema.data && parsed !== null) {
      try {
        const Ajv = (await import("ajv")).default;
        const ajv = new Ajv({ allErrors: true, strict: false });
        const validate = ajv.compile(schema.data);
        if (!validate(parsed)) {
          for (const e of validate.errors ?? []) {
            errs.push({
              path: e.instancePath || "$",
              message: e.message ?? "invalid",
            });
          }
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : "schema check failed";
        errs.push({ path: "$schema", message: msg });
      }
    }
    setErrors(errs);
    if (errs.length === 0) toast.success("Configuration valid");
    else toast.error(`${errs.length} validation error${errs.length === 1 ? "" : "s"}`);
    return errs.length === 0;
  }

  const apply = useMutation({
    mutationFn: () => applyConfig(draft),
    onSuccess: (next) => {
      toast.success("Configuration applied");
      qc.setQueryData(["config"], next);
      setConfirmOpen(false);
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : "Apply failed";
      toast.error(msg);
    },
  });

  if (live.isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-9 w-48" />
        <Skeleton className="h-[60vh] w-full" />
      </div>
    );
  }
  if (live.isError || !live.data) {
    return (
      <p className="rounded-md border border-dashed p-4 text-sm text-destructive">
        Failed to load configuration.
      </p>
    );
  }

  return (
    <div className="space-y-4" data-testid="settings-view">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle>Runtime configuration</CardTitle>
            <p className="text-xs text-muted-foreground">
              {live.data.updated_at
                ? `Last applied ${live.data.updated_at}`
                : "Live config"}
              {schema.data
                ? " · schema-validated"
                : " · schema unavailable, server-side checks only"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => void runValidate()}
              data-testid="validate-button"
            >
              <ShieldCheck className="mr-2 h-4 w-4" /> Validate
            </Button>
            <Button
              type="button"
              onClick={() => setConfirmOpen(true)}
              disabled={!dirty || apply.isPending}
              data-testid="apply-button"
            >
              <Save className="mr-2 h-4 w-4" />
              {apply.isPending ? "Applying…" : "Apply"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="edit">
            <TabsList>
              <TabsTrigger value="edit">Edit</TabsTrigger>
              <TabsTrigger value="diff" data-testid="diff-tab">
                Diff{dirty ? " ●" : ""}
              </TabsTrigger>
            </TabsList>
            <TabsContent value="edit">
              <YamlEditor value={draft} onChange={setDraft} />
              {errors.length > 0 && (
                <ul
                  className="mt-3 space-y-1 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-xs text-destructive"
                  data-testid="validation-errors"
                  aria-live="polite"
                >
                  {errors.map((e, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
                      <span>
                        <code className="font-mono">{e.path}</code> — {e.message}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              {errors.length === 0 && dirty && (
                <p className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  No validation errors.
                </p>
              )}
            </TabsContent>
            <TabsContent value="diff">
              <ConfigDiff before={live.data.yaml} after={draft} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {confirmOpen && (
        <ConfirmModal
          onCancel={() => setConfirmOpen(false)}
          onConfirm={async () => {
            const ok = await runValidate();
            if (!ok) return;
            apply.mutate();
          }}
          pending={apply.isPending}
        />
      )}
    </div>
  );
}

function ConfirmModal({
  onCancel,
  onConfirm,
  pending,
}: {
  onCancel: () => void;
  onConfirm: () => void;
  pending: boolean;
}) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-title"
      className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4"
      data-testid="apply-confirm"
    >
      <div className="w-full max-w-md space-y-4 rounded-lg border bg-background p-5 shadow-xl">
        <h3 id="confirm-title" className="text-lg font-semibold">
          Apply configuration?
        </h3>
        <p className="text-sm text-muted-foreground">
          This replaces the live runtime configuration. Validation will run
          again before the request is sent.
        </p>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={onCancel} disabled={pending}>
            Cancel
          </Button>
          <Button onClick={onConfirm} disabled={pending} data-testid="apply-confirm-button">
            {pending ? "Applying…" : "Apply"}
          </Button>
        </div>
      </div>
    </div>
  );
}
