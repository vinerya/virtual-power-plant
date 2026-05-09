"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

const MonacoEditor = dynamic(
  async () => {
    const mod = await import("@monaco-editor/react");
    return mod.default;
  },
  {
    ssr: false,
    loading: () => <Skeleton className="h-full min-h-[400px] w-full" />,
  },
);

export function YamlEditor({
  value,
  onChange,
  ariaLabel = "YAML configuration editor",
  height = "60vh",
}: {
  value: string;
  onChange: (v: string) => void;
  ariaLabel?: string;
  height?: string;
}) {
  return (
    <div
      className="overflow-hidden rounded-md border"
      role="region"
      aria-label={ariaLabel}
    >
      <MonacoEditor
        height={height}
        language="yaml"
        value={value}
        onChange={(v) => onChange(v ?? "")}
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
