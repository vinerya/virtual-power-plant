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
import { apiErrorMessage, getUrdbImportStatus, importUrdb } from "@/lib/api/tariffs";
import type { Tariff } from "@/lib/api/tariffs";

const fieldLabel = "text-xs font-medium text-muted-foreground";

/** Import a tariff from OpenEI URDB (admin; needs OPENEI_API_KEY on the server). */
export function UrdbImport({
  open,
  onOpenChange,
  onImported,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: (t: Tariff) => void;
}) {
  const qc = useQueryClient();
  const [label, setLabel] = useState("");
  const [nameOverride, setNameOverride] = useState("");

  useEffect(() => {
    if (open) {
      setLabel("");
      setNameOverride("");
    }
  }, [open]);

  const statusQ = useQuery({
    queryKey: ["tariffs", "urdb-status"],
    queryFn: getUrdbImportStatus,
    enabled: open,
    retry: false,
  });
  const configured = statusQ.data?.configured ?? false;

  const imp = useMutation({
    mutationFn: () => importUrdb(label.trim(), nameOverride.trim()),
    onSuccess: (t) => {
      toast.success(`Imported ${t.name}`);
      qc.invalidateQueries({ queryKey: ["tariffs"] });
      qc.setQueryData(["tariff", t.id], t);
      onImported(t);
    },
    onError: (e) => toast.error(apiErrorMessage(e, "Import failed")),
  });

  const error = imp.isError ? apiErrorMessage(imp.error, "Import failed") : null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent aria-label="Import from URDB">
        <SheetHeader>
          <SheetTitle>Import from OpenEI URDB</SheetTitle>
          <SheetDescription>
            Paste a record id (the <code>getpage</code> value in a{" "}
            <a
              className="underline"
              href="https://apps.openei.org/USURDB/"
              target="_blank"
              rel="noreferrer"
            >
              URDB
            </a>{" "}
            rate URL, e.g. <code>5b3104b95457a3f7437a9b2d</code>).
          </SheetDescription>
        </SheetHeader>
        {statusQ.isLoading ? (
          <p className="text-sm text-muted-foreground">Checking server configuration…</p>
        ) : !configured ? (
          <div
            role="status"
            data-testid="urdb-not-configured"
            className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm"
          >
            <p className="font-medium">URDB import is not configured on this server.</p>
            <p className="mt-1 text-muted-foreground">
              {statusQ.data?.detail ??
                "Set OPENEI_API_KEY on the API server to import tariffs from OpenEI URDB."}{" "}
              Keys are free at{" "}
              <a
                className="underline"
                href="https://openei.org/services/api/signup/"
                target="_blank"
                rel="noreferrer"
              >
                openei.org
              </a>
              . Meanwhile you can create a tariff from a preset or paste URDB JSON.
            </p>
          </div>
        ) : null}
        <form
          className="space-y-3"
          data-testid="urdb-import-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (label.trim()) imp.mutate();
          }}
        >
          <div>
            <label htmlFor="urdb-label" className={fieldLabel}>
              URDB record id
            </label>
            <Input
              id="urdb-label"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              disabled={!configured}
              className="mt-1 font-mono"
            />
          </div>
          <div>
            <label htmlFor="urdb-name" className={fieldLabel}>
              Name (optional)
            </label>
            <Input
              id="urdb-name"
              value={nameOverride}
              onChange={(e) => setNameOverride(e.target.value)}
              disabled={!configured}
              className="mt-1"
            />
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
              Close
            </Button>
            <Button type="submit" disabled={!configured || !label.trim() || imp.isPending}>
              {imp.isPending ? "Importing…" : "Import"}
            </Button>
          </div>
        </form>
      </SheetContent>
    </Sheet>
  );
}
