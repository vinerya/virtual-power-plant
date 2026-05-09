"use client";

import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Copy } from "lucide-react";
import { toast } from "sonner";
import { copyToClipboard } from "@/lib/utils";
import type { Bill } from "@/lib/api/types";

const KIND_COLORS: Record<string, string> = {
  fixed: "#94a3b8",
  energy: "hsl(var(--primary))",
  demand: "#f59e0b",
  min_bill: "#a78bfa",
  credit: "#10b981",
};

export function BillBreakdown({
  bill,
  title,
  comparison,
}: {
  bill: Bill;
  title?: string;
  comparison?: Bill | null;
}) {
  const data = bill.line_items.map((li) => ({
    name: li.name,
    amount: li.amount,
    kind: li.kind,
  }));
  const currency = bill.currency ?? "USD";

  return (
    <section
      className="space-y-3 rounded-md border p-4"
      aria-label={`Bill breakdown${title ? ` ${title}` : ""}`}
      data-testid="bill-breakdown"
    >
      <header className="flex items-center justify-between">
        <h4 className="text-sm font-semibold">{title ?? "Bill"}</h4>
        <div className="flex items-center gap-2">
          {comparison && (
            <Badge variant="outline" className="text-xs">
              vs {comparison.tariff_id}: $
              {(bill.total - comparison.total).toFixed(2)}
            </Badge>
          )}
          <Button
            type="button"
            variant="outline"
            size="sm"
            aria-label="Copy bill JSON"
            onClick={() => {
              copyToClipboard(JSON.stringify(bill, null, 2))
                .then(() => toast.success("Bill JSON copied"))
                .catch(() => toast.error("Clipboard unavailable"));
            }}
          >
            <Copy className="mr-1 h-3.5 w-3.5" />
            Copy
          </Button>
        </div>
      </header>

      <div className="h-40 w-full" role="img" aria-label="Stacked bill components">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={[{ name: "Total", ...flatten(data) }]}
            margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
            stackOffset="sign"
          >
            <XAxis
              type="number"
              tickFormatter={(v: number) => `$${v.toFixed(0)}`}
              fontSize={11}
            />
            <YAxis type="category" dataKey="name" hide />
            <Tooltip
              contentStyle={{
                background: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(v: number, n: string) => [`$${v.toFixed(2)}`, n]}
            />
            {data.map((d, i) => (
              <Bar key={i} dataKey={d.name} stackId="a" isAnimationActive={false}>
                <Cell fill={KIND_COLORS[d.kind] ?? "#cbd5e1"} />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <table className="w-full text-sm">
        <tbody>
          {bill.line_items.map((li, i) => (
            <tr key={i} className="border-b last:border-b-0">
              <td className="py-1.5">
                <span
                  aria-hidden="true"
                  className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                  style={{ background: KIND_COLORS[li.kind] ?? "#cbd5e1" }}
                />
                {li.name}
                <span className="ml-2 text-xs text-muted-foreground">
                  {li.kind}
                </span>
              </td>
              <td className="py-1.5 text-right tabular-nums">
                ${li.amount.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="border-t font-semibold">
            <td className="pt-2">Total ({currency})</td>
            <td className="pt-2 text-right tabular-nums" data-testid="bill-total">
              ${bill.total.toFixed(2)}
            </td>
          </tr>
        </tfoot>
      </table>
    </section>
  );
}

function flatten(rows: { name: string; amount: number }[]) {
  const out: Record<string, number> = {};
  for (const r of rows) out[r.name] = r.amount;
  return out;
}
