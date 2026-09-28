"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  apiErrorMessage,
  createTariff,
  getTariffPreset,
  listTariffPresets,
  updateTariff,
} from "@/lib/api/tariffs";
import type { Tariff, TariffWrite } from "@/lib/api/tariffs";
import { NEM_LABELS } from "./format";

const fieldLabel = "text-xs font-medium text-muted-foreground";

/**
 * Create a tariff (optionally from a bundled preset) or edit an existing
 * one. The URDB JSON is the source of truth; the server validates that the
 * bill engine can evaluate it before saving.
 */
export function TariffEditor({
  open,
  tariff,
  onOpenChange,
  onSaved,
}: {
  open: boolean;
  /** null = create */
  tariff: Tariff | null;
  onOpenChange: (open: boolean) => void;
  onSaved: (t: Tariff) => void;
}) {
  const qc = useQueryClient();
  const [presetId, setPresetId] = useState("");
  const [name, setName] = useState("");
  const [utility, setUtility] = useState("");
  const [effective, setEffective] = useState("");
  const [json, setJson] = useState("{}");
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setPresetId("");
    setJsonError(null);
    setName(tariff?.name ?? "");
    setUtility(tariff?.utility ?? "");
    setEffective(tariff?.effective_date ?? "");
    setJson(JSON.stringify(tariff?.urdb_json ?? {}, null, 2));
  }, [open, tariff]);

  const presets = useQuery({
    queryKey: ["tariff-presets"],
    queryFn: listTariffPresets,
    enabled: open && !tariff,
    staleTime: 10 * 60_000,
  });

  const loadPreset = useMutation({
    mutationFn: (id: string) => getTariffPreset(id),
    onSuccess: (p) => {
      setName(p.name);
      setUtility(p.utility ?? "");
      setEffective(typeof p.urdb_json.startdate === "string" ? p.urdb_json.startdate : "");
      setJson(JSON.stringify(p.urdb_json, null, 2));
      setJsonError(null);
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Failed to load preset")),
  });

  const parsed = (): Record<string, unknown> | null => {
    try {
      const v = JSON.parse(json);
      if (!v || typeof v !== "object" || Array.isArray(v)) {
        setJsonError("URDB JSON must be an object");
        return null;
      }
      setJsonError(null);
      return v as Record<string, unknown>;
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : "Invalid JSON");
      return null;
    }
  };

  const nemValue = (() => {
    try {
      const v = JSON.parse(json) as Record<string, unknown>;
      return typeof v.nem === "string" ? v.nem : "";
    } catch {
      return "";
    }
  })();

  const setNem = (value: string) => {
    const v = parsed();
    if (!v) return;
    if (value) v.nem = value;
    else delete v.nem;
    setJson(JSON.stringify(v, null, 2));
  };

  const save = useMutation({
    mutationFn: (body: TariffWrite) =>
      tariff ? updateTariff(tariff.id, body) : createTariff(body),
    onSuccess: (t) => {
      toast.success(tariff ? "Tariff updated" : "Tariff created");
      qc.invalidateQueries({ queryKey: ["tariffs"] });
      qc.setQueryData(["tariff", t.id], t);
      onSaved(t);
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Save failed")),
  });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const urdb = parsed();
    if (!urdb || !name.trim()) return;
    save.mutate({
      name: name.trim(),
      utility: utility.trim(),
      urdb_json: urdb,
      effective_date: effective || null,
    });
  };

  const saveError = save.isError ? apiErrorMessage(save.error, "Save failed") : null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent aria-label={tariff ? "Edit tariff" : "New tariff"} className="max-w-2xl">
        <SheetHeader>
          <SheetTitle>{tariff ? `Edit ${tariff.name}` : "New tariff"}</SheetTitle>
          <SheetDescription>
            Tariffs are stored as{" "}
            <a
              className="underline"
              href="https://openei.org/services/doc/rest/util_rates/?version=8"
              target="_blank"
              rel="noreferrer"
            >
              OpenEI URDB
            </a>{" "}
            JSON. Extensions: <code>adders</code>, <code>taxes</code>, <code>nem</code>,{" "}
            <code>nem3_avoided_cost</code>.
          </SheetDescription>
        </SheetHeader>
        <form className="space-y-3" onSubmit={submit} data-testid="tariff-editor">
          {!tariff && (
            <div>
              <label htmlFor="tariff-preset" className={fieldLabel}>
                Start from preset
              </label>
              <select
                id="tariff-preset"
                value={presetId}
                onChange={(e) => {
                  setPresetId(e.target.value);
                  if (e.target.value) loadPreset.mutate(e.target.value);
                }}
                className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
              >
                <option value="">— blank —</option>
                {(presets.data ?? []).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                    {p.illustrative ? " (illustrative)" : ""}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label htmlFor="tariff-name" className={fieldLabel}>
                Name
              </label>
              <Input
                id="tariff-name"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <label htmlFor="tariff-utility" className={fieldLabel}>
                Utility
              </label>
              <Input
                id="tariff-utility"
                value={utility}
                onChange={(e) => setUtility(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <label htmlFor="tariff-effective" className={fieldLabel}>
                Effective date
              </label>
              <Input
                id="tariff-effective"
                type="date"
                value={effective}
                onChange={(e) => setEffective(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <label htmlFor="tariff-nem" className={fieldLabel}>
                Export credit (NEM)
              </label>
              <select
                id="tariff-nem"
                value={nemValue}
                onChange={(e) => setNem(e.target.value)}
                className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
              >
                <option value="">From URDB dgrules / none</option>
                {Object.entries(NEM_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div>
            <label htmlFor="tariff-json" className={fieldLabel}>
              URDB JSON
            </label>
            <textarea
              id="tariff-json"
              value={json}
              onChange={(e) => setJson(e.target.value)}
              onBlur={() => parsed()}
              spellCheck={false}
              rows={16}
              className="mt-1 w-full rounded-md border border-input bg-background p-2 font-mono text-xs"
              aria-invalid={jsonError ? true : undefined}
              aria-describedby={jsonError ? "tariff-json-error" : undefined}
            />
            {jsonError && (
              <p id="tariff-json-error" role="alert" className="text-xs text-destructive">
                {jsonError}
              </p>
            )}
          </div>
          {saveError && (
            <p role="alert" className="text-sm text-destructive" data-testid="tariff-save-error">
              {saveError}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={save.isPending || !name.trim()}>
              {save.isPending ? "Saving…" : tariff ? "Save changes" : "Create tariff"}
            </Button>
          </div>
        </form>
      </SheetContent>
    </Sheet>
  );
}
