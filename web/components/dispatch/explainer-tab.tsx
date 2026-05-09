"use client";

import dynamic from "next/dynamic";
import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { getExplainer } from "@/lib/api/explainer";
import type { ExplainerResponse } from "@/lib/api/types";

const CostComparisonChart = dynamic(
  () =>
    import("./cost-comparison-chart").then((m) => m.CostComparisonChart),
  { ssr: false, loading: () => <Skeleton className="h-56 w-full" /> },
);

const PerStepComparisonChart = dynamic(
  () =>
    import("./per-step-comparison-chart").then(
      (m) => m.PerStepComparisonChart,
    ),
  { ssr: false, loading: () => <Skeleton className="h-64 w-full" /> },
);

export function ExplainerTab({
  runId,
  rationaleFallback,
}: {
  runId: string;
  rationaleFallback?: string;
}) {
  const q = useQuery({
    queryKey: ["explainer", runId],
    queryFn: () => getExplainer(runId),
    staleTime: 30_000,
  });

  if (q.isLoading) {
    return (
      <div className="space-y-3" data-testid="explainer-loading">
        <Skeleton className="h-56 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (q.isError) {
    return (
      <p className="rounded-md border border-dashed p-4 text-sm text-destructive">
        Failed to load explainer data.
      </p>
    );
  }

  const data = q.data ?? null;
  if (!data) {
    return (
      <div
        data-testid="explainer-empty"
        className="rounded-md border border-dashed p-4 text-sm text-muted-foreground"
      >
        Explainer data not available for this run.
        {rationaleFallback && (
          <p className="mt-2 text-foreground">{rationaleFallback}</p>
        )}
      </div>
    );
  }

  return <ExplainerBody data={data} rationaleFallback={rationaleFallback} />;
}

function ExplainerBody({
  data,
  rationaleFallback,
}: {
  data: ExplainerResponse;
  rationaleFallback?: string;
}) {
  const baseline =
    data.counterfactuals.find((c) => c.name === "no_action") ??
    data.counterfactuals[0];
  const savings = baseline ? baseline.total_cost - data.actual.total_cost : 0;

  return (
    <section className="space-y-5" data-testid="explainer-content">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Counterfactual comparison</h3>
          <p className="text-xs text-muted-foreground">
            Realized dispatch vs. {data.counterfactuals.length} baseline
            {data.counterfactuals.length === 1 ? "" : "s"}.
          </p>
        </div>
        {baseline && (
          <p
            className={`text-sm tabular-nums ${
              savings >= 0 ? "text-emerald-600" : "text-destructive"
            }`}
          >
            {savings >= 0 ? "Saved" : "Cost"} ${Math.abs(savings).toFixed(2)} vs{" "}
            <span className="font-mono">{baseline.name}</span>
          </p>
        )}
      </header>

      <CostComparisonChart
        actual={data.actual}
        counterfactuals={data.counterfactuals}
      />

      <div>
        <h4 className="mb-1 text-sm font-semibold">Per-step net power</h4>
        <PerStepComparisonChart
          actual={data.actual}
          counterfactuals={data.counterfactuals}
        />
      </div>

      {(data.rationale || rationaleFallback) && (
        <div>
          <h4 className="text-sm font-semibold">Rationale</h4>
          <p className="mt-1 text-sm text-muted-foreground">
            {data.rationale ?? rationaleFallback}
          </p>
        </div>
      )}

      {data.binding_constraints.length > 0 && (
        <div data-testid="binding-constraints">
          <h4 className="text-sm font-semibold">Binding constraints</h4>
          <ul className="mt-2 space-y-1 text-sm">
            {data.binding_constraints.map((c, i) => (
              <li
                key={i}
                className="flex items-center justify-between rounded-md border bg-muted/30 px-3 py-1.5"
              >
                <span>
                  <Badge variant="outline" className="mr-2 font-mono text-[10px]">
                    t={c.step}
                  </Badge>
                  {c.description ?? c.name}
                </span>
                <span className="text-xs tabular-nums text-muted-foreground">
                  slack {c.slack.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
