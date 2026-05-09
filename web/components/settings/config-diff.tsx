"use client";

import { useMemo } from "react";

/**
 * Hand-rolled line-by-line diff. Algorithm: longest-common-subsequence
 * over lines, then walk the table to emit "same" / "del" / "add" rows.
 *
 * Good enough for short YAML configs (< few thousand lines). If we
 * outgrow it, swap in `diff` or `react-diff-view`.
 */
type Op = "same" | "add" | "del";
interface DiffRow {
  left?: { n: number; text: string };
  right?: { n: number; text: string };
  op: Op;
}

function diffLines(a: string[], b: string[]): DiffRow[] {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0),
  );
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (a[i] === b[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out: DiffRow[] = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (a[i] === b[j]) {
      out.push({
        op: "same",
        left: { n: i + 1, text: a[i] },
        right: { n: j + 1, text: b[j] },
      });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ op: "del", left: { n: i + 1, text: a[i] } });
      i++;
    } else {
      out.push({ op: "add", right: { n: j + 1, text: b[j] } });
      j++;
    }
  }
  while (i < m) out.push({ op: "del", left: { n: i + 1, text: a[i++] } });
  while (j < n) out.push({ op: "add", right: { n: j + 1, text: b[j++] } });
  return out;
}

export function ConfigDiff({
  before,
  after,
}: {
  before: string;
  after: string;
}) {
  const rows = useMemo(
    () => diffLines(before.split("\n"), after.split("\n")),
    [before, after],
  );

  const hasChanges = rows.some((r) => r.op !== "same");
  if (!hasChanges) {
    return (
      <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
        No unsaved changes.
      </p>
    );
  }

  return (
    <div
      className="overflow-x-auto rounded-md border bg-card font-mono text-xs"
      role="region"
      aria-label="Configuration diff"
      data-testid="config-diff"
    >
      <table className="w-full">
        <tbody>
          {rows.map((r, idx) => (
            <tr key={idx} className="align-top">
              <td className="w-10 select-none border-r bg-muted/40 px-2 py-0.5 text-right text-muted-foreground">
                {r.left?.n ?? ""}
              </td>
              <td
                className={cellClass("left", r.op)}
                data-op={r.op === "del" ? "del" : "same"}
              >
                {r.left?.text ?? ""}
              </td>
              <td className="w-10 select-none border-r border-l bg-muted/40 px-2 py-0.5 text-right text-muted-foreground">
                {r.right?.n ?? ""}
              </td>
              <td
                className={cellClass("right", r.op)}
                data-op={r.op === "add" ? "add" : "same"}
              >
                {r.right?.text ?? ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function cellClass(side: "left" | "right", op: Op): string {
  const base = "whitespace-pre px-2 py-0.5";
  if (op === "same") return base;
  if (side === "left" && op === "del")
    return `${base} bg-red-500/15 text-red-700 dark:text-red-300`;
  if (side === "right" && op === "add")
    return `${base} bg-emerald-500/15 text-emerald-700 dark:text-emerald-300`;
  return `${base} text-muted-foreground/40`;
}
