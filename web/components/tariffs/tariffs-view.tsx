"use client";

import { useCallback, useState } from "react";
import type { Tariff } from "@/lib/api/tariffs";
import { TariffList } from "./tariff-list";
import { TariffDetail } from "./tariff-detail";
import { TariffEditor } from "./tariff-editor";
import { UrdbImport } from "./urdb-import";
import { useIsAdmin } from "./use-is-admin";

type EditorState = { mode: "create" } | { mode: "edit"; tariff: Tariff } | null;

export function TariffsView({ initialId = null }: { initialId?: string | null }) {
  const [selected, setSelectedState] = useState<string | null>(initialId);
  const [editor, setEditor] = useState<EditorState>(null);
  const [importOpen, setImportOpen] = useState(false);
  const isAdmin = useIsAdmin();

  const setSelected = useCallback((id: string | null) => {
    setSelectedState(id);
    // Keep the URL deep-linkable without a navigation round-trip.
    if (typeof window !== "undefined") {
      window.history.replaceState(null, "", id ? `/tariffs/${encodeURIComponent(id)}` : "/tariffs");
    }
  }, []);

  return (
    <div
      className="flex h-[calc(100vh-7rem)] overflow-hidden rounded-md border bg-card"
      data-testid="tariffs-view"
    >
      <TariffList
        selectedId={selected}
        onSelect={setSelected}
        onNew={isAdmin ? () => setEditor({ mode: "create" }) : undefined}
        onImport={isAdmin ? () => setImportOpen(true) : undefined}
      />
      <TariffDetail
        tariffId={selected}
        isAdmin={isAdmin}
        onEdit={(t) => setEditor({ mode: "edit", tariff: t })}
        onDeleted={() => setSelected(null)}
      />
      <TariffEditor
        open={editor !== null}
        tariff={editor?.mode === "edit" ? editor.tariff : null}
        onOpenChange={(o) => !o && setEditor(null)}
        onSaved={(t) => {
          setEditor(null);
          setSelected(t.id);
        }}
      />
      <UrdbImport
        open={importOpen}
        onOpenChange={setImportOpen}
        onImported={(t) => {
          setImportOpen(false);
          setSelected(t.id);
        }}
      />
    </div>
  );
}
