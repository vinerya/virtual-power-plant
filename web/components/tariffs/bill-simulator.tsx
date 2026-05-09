"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Play } from "lucide-react";
import { listTariffs, simulateBill } from "@/lib/api/tariffs";
import type { SimulateRequest } from "@/lib/api/tariffs";
import { BillBreakdown } from "./bill-breakdown";

export function BillSimulator({ tariffId }: { tariffId: string }) {
  const [synthetic, setSynthetic] = useState(true);
  const [csv, setCsv] = useState<string | null>(null);
  const [csvName, setCsvName] = useState<string | null>(null);
  const [compareTo, setCompareTo] = useState<string>("");

  const tariffs = useQuery({
    queryKey: ["tariffs"],
    queryFn: listTariffs,
    staleTime: 60_000,
  });

  const sim = useMutation({
    mutationFn: () => {
      const req: SimulateRequest = synthetic
        ? { synthetic: true, period_days: 30, compare_to: compareTo || undefined }
        : { csv: csv ?? "", compare_to: compareTo || undefined };
      return simulateBill(tariffId, req);
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : "Simulation failed";
      toast.error(msg);
    },
    onSuccess: () => toast.success("Simulation complete"),
  });

  const onFile = (f: File | null) => {
    if (!f) {
      setCsv(null);
      setCsvName(null);
      return;
    }
    setCsvName(f.name);
    const reader = new FileReader();
    reader.onload = () => setCsv(String(reader.result ?? ""));
    reader.readAsText(f);
    setSynthetic(false);
  };

  return (
    <div className="space-y-4" data-testid="bill-simulator">
      <fieldset className="space-y-3 rounded-md border p-4">
        <legend className="px-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Load profile
        </legend>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={synthetic}
            onChange={(e) => setSynthetic(e.target.checked)}
            data-testid="synthetic-toggle"
          />
          Use synthetic 30-day load
        </label>
        <div>
          <label
            htmlFor="csv-upload"
            className="text-xs font-medium text-muted-foreground"
          >
            Or upload meter trace CSV
          </label>
          <input
            id="csv-upload"
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            className="mt-1 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-3 file:py-1.5 file:text-sm hover:file:bg-muted/80"
            aria-describedby="csv-help"
          />
          <p id="csv-help" className="mt-1 text-xs text-muted-foreground">
            Expected columns: <code>timestamp,kw</code>. {csvName ?? "No file chosen."}
          </p>
        </div>
        <div>
          <label
            htmlFor="compare-to"
            className="text-xs font-medium text-muted-foreground"
          >
            Compare to (optional)
          </label>
          <select
            id="compare-to"
            value={compareTo}
            onChange={(e) => setCompareTo(e.target.value)}
            className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
          >
            <option value="">— none —</option>
            {(tariffs.data ?? [])
              .filter((t) => t.id !== tariffId)
              .map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
          </select>
        </div>
        <Button
          type="button"
          onClick={() => sim.mutate()}
          disabled={sim.isPending || (!synthetic && !csv)}
          data-testid="run-simulation"
        >
          <Play className="mr-2 h-3.5 w-3.5" />
          {sim.isPending ? "Simulating…" : "Run simulation"}
        </Button>
      </fieldset>

      {sim.isPending && <Skeleton className="h-48 w-full" />}
      {sim.data?.bill && (
        <div className="grid gap-4 md:grid-cols-2">
          <BillBreakdown bill={sim.data.bill} title="Selected tariff" />
          {sim.data.comparison && (
            <BillBreakdown
              bill={sim.data.comparison}
              title="Comparison"
              comparison={sim.data.bill}
            />
          )}
        </div>
      )}
    </div>
  );
}
