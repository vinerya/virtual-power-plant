"use client";

import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AreaScheduleChart } from "@/components/charts/lazy-line-chart";
import type { DispatchRun } from "@/lib/api/types";
import { formatDateTime } from "@/lib/utils";

export function DispatchSheet({
  run,
  onClose,
}: {
  run: DispatchRun | null;
  onClose: () => void;
}) {
  const open = run !== null;
  const schedule = useMemo(() => extractSchedule(run), [run]);
  const pricePreview = useMemo(() => extractPricePreview(run), [run]);

  return (
    <Sheet open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent aria-label="Dispatch detail" data-testid="dispatch-sheet">
        {run && (
          <>
            <SheetHeader>
              <SheetTitle>{run.problem_type} dispatch</SheetTitle>
              <p className="text-sm text-muted-foreground">
                {formatDateTime(run.created_at)}
                {run.fallback_used && (
                  <Badge className="ml-2" variant="secondary">
                    fallback
                  </Badge>
                )}
              </p>
            </SheetHeader>

            <Tabs defaultValue="solution" className="mt-2">
              <TabsList>
                <TabsTrigger value="inputs">Inputs</TabsTrigger>
                <TabsTrigger value="solution">Solution</TabsTrigger>
                <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
                <TabsTrigger value="counterfactual">Counterfactual</TabsTrigger>
              </TabsList>

              <TabsContent value="inputs">
                <section className="space-y-4">
                  <div>
                    <h3 className="text-sm font-semibold">Price preview</h3>
                    {pricePreview.length > 0 ? (
                      <ul className="mt-2 grid grid-cols-4 gap-2 text-xs tabular-nums sm:grid-cols-6">
                        {pricePreview.map((p, i) => (
                          <li
                            key={i}
                            className="rounded border bg-muted/40 px-2 py-1 text-center"
                          >
                            {p.toFixed(2)}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="mt-2 text-sm text-muted-foreground">
                        No price vector recorded.
                      </p>
                    )}
                  </div>
                  <KVList
                    title="Key parameters"
                    obj={(run.inputs as Record<string, unknown> | undefined) ?? {}}
                  />
                </section>
              </TabsContent>

              <TabsContent value="solution">
                <section className="space-y-4" data-testid="solution-section">
                  {schedule.length > 0 ? (
                    <AreaScheduleChart data={schedule} />
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No per-timestep schedule recorded.
                    </p>
                  )}
                  <KVList
                    title="Solution summary"
                    obj={summarizeSolution(run.solution)}
                  />
                </section>
              </TabsContent>

              <TabsContent value="diagnostics">
                <section className="space-y-3 text-sm">
                  <KVList
                    title="Solver"
                    obj={{
                      solver: run.solver ?? "—",
                      iterations: run.iterations ?? "—",
                      gap: run.gap ?? "—",
                      solve_time_ms: run.solve_time_ms ?? "—",
                      status: run.status,
                    }}
                  />
                  {(run.metadata as { rationale?: string } | undefined)
                    ?.rationale && (
                    <div>
                      <h3 className="text-sm font-semibold">Rationale</h3>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {(run.metadata as { rationale: string }).rationale}
                      </p>
                    </div>
                  )}
                </section>
              </TabsContent>

              <TabsContent value="counterfactual">
                <section className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Coming with M3 — the dispatch explainer will compare this run
                  against a no-action and a price-naive baseline.
                </section>
              </TabsContent>
            </Tabs>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

function KVList({
  title,
  obj,
}: {
  title: string;
  obj: Record<string, unknown>;
}) {
  const rows = Object.entries(obj).filter(
    ([, v]) =>
      v !== undefined &&
      v !== null &&
      (typeof v !== "object" || Array.isArray(v)),
  );
  if (rows.length === 0) return null;
  return (
    <div>
      <h3 className="text-sm font-semibold">{title}</h3>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="tabular-nums">
              {Array.isArray(v)
                ? v.length > 8
                  ? `[${v.slice(0, 8).join(", ")}, …]`
                  : `[${v.join(", ")}]`
                : String(v)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function extractPricePreview(run: DispatchRun | null): number[] {
  if (!run) return [];
  const inputs = run.inputs as Record<string, unknown> | undefined;
  const candidates = [
    inputs?.prices,
    inputs?.price_vector,
    inputs?.base_prices,
  ];
  for (const c of candidates) {
    if (Array.isArray(c) && c.every((x) => typeof x === "number")) {
      return (c as number[]).slice(0, 24);
    }
  }
  return [];
}

function extractSchedule(run: DispatchRun | null) {
  if (!run?.solution) return [];
  const sol = run.solution as Record<string, unknown>;
  const charge = numericArray(sol.charge);
  const discharge = numericArray(sol.discharge);
  const power = numericArray(sol.power) ?? numericArray(sol.schedule);
  const len = Math.max(
    charge?.length ?? 0,
    discharge?.length ?? 0,
    power?.length ?? 0,
  );
  if (len === 0) return [];
  const out = [] as { step: number; charge?: number; discharge?: number; power?: number }[];
  for (let i = 0; i < len; i++) {
    out.push({
      step: i,
      charge: charge?.[i],
      discharge: discharge?.[i] != null ? -Math.abs(discharge[i]) : undefined,
      power: power?.[i],
    });
  }
  return out;
}

function numericArray(v: unknown): number[] | null {
  if (Array.isArray(v) && v.every((x) => typeof x === "number")) {
    return v as number[];
  }
  return null;
}

function summarizeSolution(s: Record<string, unknown> | undefined) {
  if (!s) return {};
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(s)) {
    if (typeof v === "number") out[k] = v;
    else if (Array.isArray(v) && v.every((x) => typeof x === "number")) {
      const arr = v as number[];
      out[`${k}.length`] = arr.length;
      out[`${k}.sum`] = Number(arr.reduce((a, b) => a + b, 0).toFixed(3));
    }
  }
  return out;
}
