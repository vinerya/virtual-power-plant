"use client";

import { useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import type { OnMount } from "@monaco-editor/react";
import { Skeleton } from "@/components/ui/skeleton";
import { pointerToLine } from "@/lib/yaml-pointer";

const MonacoEditor = dynamic(
  async () => {
    const mod = await import("@monaco-editor/react");
    // Serve Monaco from our own origin (copied to public/monaco by
    // scripts/copy-monaco.mjs) rather than the default jsDelivr CDN.
    mod.loader.config({ paths: { vs: "/monaco/vs" } });
    return mod.default;
  },
  {
    ssr: false,
    loading: () => <Skeleton className="h-full min-h-[400px] w-full" />,
  },
);

export interface EditorMarker {
  /** JSON-Pointer path ("/optimizer/solver") or "$" for the document. */
  path: string;
  message: string;
}

type Editor = Parameters<OnMount>[0];
type Monaco = Parameters<OnMount>[1];

const MARKER_OWNER = "vpp-config";

export function YamlEditor({
  value,
  onChange,
  ariaLabel = "YAML configuration editor",
  height = "60vh",
  markers = [],
}: {
  value: string;
  onChange: (v: string) => void;
  ariaLabel?: string;
  height?: string;
  /** Validation errors to underline inline (resolved to YAML lines). */
  markers?: EditorMarker[];
}) {
  const editorRef = useRef<Editor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);

  const apply = () => {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    const model = editor?.getModel();
    if (!editor || !monaco || !model) return;
    const text = model.getValue();
    monaco.editor.setModelMarkers(
      model,
      MARKER_OWNER,
      markers.map((m) => {
        const line = Math.min(pointerToLine(text, m.path), model.getLineCount());
        return {
          severity: monaco.MarkerSeverity.Error,
          message: m.path && m.path !== "$" ? `${m.path}: ${m.message}` : m.message,
          startLineNumber: line,
          startColumn: model.getLineFirstNonWhitespaceColumn(line) || 1,
          endLineNumber: line,
          endColumn: model.getLineMaxColumn(line),
        };
      }),
    );
  };

  // Re-place markers whenever they change. Edits don't clear them: they are
  // cleared by the parent when the draft is re-validated.
  useEffect(apply, [markers]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      className="overflow-hidden rounded-md border"
      role="region"
      aria-label={ariaLabel}
      data-testid="yaml-editor"
      data-marker-count={markers.length}
    >
      <MonacoEditor
        height={height}
        language="yaml"
        value={value}
        onChange={(v) => onChange(v ?? "")}
        onMount={(editor, monaco) => {
          editorRef.current = editor;
          monacoRef.current = monaco;
          apply();
        }}
        theme="vs-dark"
        options={{
          minimap: { enabled: false },
          fontSize: 13,
          lineNumbers: "on",
          tabSize: 2,
          scrollBeyondLastLine: false,
          automaticLayout: true,
          ariaLabel,
        }}
      />
    </div>
  );
}
