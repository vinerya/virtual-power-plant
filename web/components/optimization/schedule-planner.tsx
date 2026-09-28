"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ExternalLink, Play, Wand2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Field, Stat } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { listResources } from "@/lib/api/resources";
import { apiErrorMessage } from "@/lib/api/errors";
import {
  INTERVALS,
  MAX_SCHEDULE_STEPS,
  listTariffOptions,
  parseSeries,
  planSchedule,
  sampleDayPrices,
  type ScheduleRequest,
  type ScheduleResponse,
} from "@/lib/api/optimization";
import { formatNumber } from "@/lib/utils";
import { PowerPriceChart, SocChart } from "./lazy-charts";

type PriceSource = "series" | "tariff";

export function SchedulePlanner() {
  const resources = useQuery({ queryKey: ["resources"], queryFn: listResources, staleTime: 60_000 });
  const batteries = useMemo(
    () => (resources.data ?? []).filter((r) => r.resource_type === "battery"),
    [resources.data],
  );
  const tariffs = useQuery({
    queryKey: ["optimization", "tariff-options"],
    queryFn: listTariffOptions,
    staleTime: 60_000,
  });

  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [source, setSource] = useState<PriceSource>("series");
  const [intervalMin, setIntervalMin] = useState(60);
  const [pricesText, setPricesText] = useState(() => sampleDayPrices(24, 60).join(", "));
  const [tariffId, setTariffId] = useState("");
  const [horizonHours, setHorizonHours] = useState("24");
  const [nem, setNem] = useState<"none" | "nem2" | "nem3">("nem2");
  const [degradation, setDegradation] = useState(true);
  const [replacementCost, setReplacementCost] = useState("250");
  const [terminal, setTerminal] = useState<"value" | "hold">("value");
  const [formError, setFormError] = useState<string | null>(null);

  const parsed = useMemo(() => parseSeries(pricesText), [pricesText]);

  const run = useMutation({ mutationFn: planSchedule });

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);
    const body: ScheduleRequest = {
      interval_minutes: intervalMin,
      degradation_aware: degradation,
      replacement_cost_per_kwh: Number(replacementCost) || 250,
      terminal_soc_policy: terminal,
    };
    if (selectedIds.length) body.resource_ids = selectedIds;
    if (source === "series") {
      if (parsed.invalid.length) {
        setFormError(`Not a number: ${parsed.invalid.slice(0, 3).join(", ")}`);
        return;
      }
      if (parsed.values.length < 1 || parsed.values.length > MAX_SCHEDULE_STEPS) {
        setFormError(`Enter between 1 and ${MAX_SCHEDULE_STEPS} prices (one per step).`);
        return;
      }
      body.prices = parsed.values;
    } else {
      if (!tariffId) {
        setFormError("Choose a tariff.");
        return;
      }
      const h = Number(horizonHours);
      if (!(h >= 1 && h <= 168) || (h * 60) / intervalMin > MAX_SCHEDULE_STEPS) {
        setFormError(
          `Horizon must be 1–168 h and at most ${MAX_SCHEDULE_STEPS} steps at this interval.`,
        );
        return;
      }
      body.tariff_id = tariffId;
      body.horizon_hours = h;
      body.nem = nem;
    }
    run.mutate(body);
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base text-foreground">Plan a horizon schedule</CardTitle>
          <p className="text-xs text-muted-foreground">
            One MPC solve over the horizon for the selected batteries, using their persisted
            capacity, SOC and state of health. Each run is saved to the dispatch history.
          </p>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4" aria-label="Schedule request">
            <div className="grid gap-4 lg:grid-cols-2">
              <Field
                label="Batteries"
                hint={
                  resources.isLoading
                    ? "Loading resources…"
                    : batteries.length === 0
                      ? "No battery resources found: add one under Assets first."
                      : "Hold Ctrl/⌘ to select several. None selected = all online batteries."
                }
              >
                <select
                  multiple
                  value={selectedIds}
                  onChange={(e) =>
                    setSelectedIds(Array.from(e.target.selectedOptions).map((o) => o.value))
                  }
                  className="h-24 w-full rounded-md border border-input bg-background px-2 py-1 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  data-testid="schedule-batteries"
                >
                  {batteries.map((b) => (
                    <option key={b.id} value={b.id} disabled={!b.online}>
                      {b.name}
                      {b.online ? "" : " (offline)"}
                    </option>
                  ))}
                </select>
              </Field>

              <fieldset className="space-y-2">
                <legend className="text-xs font-medium text-muted-foreground">Price source</legend>
                <div className="flex gap-4 text-sm">
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="price-source"
                      checked={source === "series"}
                      onChange={() => setSource("series")}
                    />
                    Price series
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="price-source"
                      checked={source === "tariff"}
                      onChange={() => setSource("tariff")}
                      data-testid="source-tariff"
                    />
                    Stored tariff
                  </label>
                </div>
                <Field label="Interval (minutes)">
                  <Select
                    value={intervalMin}
                    onChange={(e) => setIntervalMin(Number(e.target.value))}
                    data-testid="schedule-interval"
                  >
                    {INTERVALS.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </Select>
                </Field>
              </fieldset>
            </div>

            {source === "series" ? (
              <Field
                label="Prices (currency per kWh, one per step)"
                hint={`${parsed.values.length} step${parsed.values.length === 1 ? "" : "s"} = ${formatNumber((parsed.values.length * intervalMin) / 60, 1)} h. Separate with commas, spaces or new lines.`}
              >
                <textarea
                  value={pricesText}
                  onChange={(e) => setPricesText(e.target.value)}
                  rows={3}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  data-testid="schedule-prices"
                />
              </Field>
            ) : (
              <div className="grid gap-3 sm:grid-cols-3">
                <Field
                  label="Tariff"
                  hint={
                    tariffs.isError
                      ? apiErrorMessage(tariffs.error, "Could not load tariffs.")
                      : tariffs.data?.length === 0
                        ? "No tariffs stored yet (see Tariffs)."
                        : undefined
                  }
                >
                  <Select
                    value={tariffId}
                    onChange={(e) => setTariffId(e.target.value)}
                    data-testid="schedule-tariff"
                  >
                    <option value="">Choose…</option>
                    {(tariffs.data ?? []).map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                        {t.utility ? ` · ${t.utility}` : ""}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Horizon (hours)" hint="Starts at the current interval">
                  <Input
                    type="number"
                    min={1}
                    max={168}
                    value={horizonHours}
                    onChange={(e) => setHorizonHours(e.target.value)}
                  />
                </Field>
                <Field label="Export compensation">
                  <Select value={nem} onChange={(e) => setNem(e.target.value as typeof nem)}>
                    <option value="nem2">NEM 2 (retail credit)</option>
                    <option value="nem3">NEM 3 (avoided cost)</option>
                    <option value="none">No export credit</option>
                  </Select>
                </Field>
              </div>
            )}

            <div className="grid gap-3 sm:grid-cols-3">
              <label className="flex items-start gap-2 text-sm">
                <input
                  type="checkbox"
                  className="mt-1"
                  checked={degradation}
                  onChange={(e) => setDegradation(e.target.checked)}
                  data-testid="schedule-degradation"
                />
                <span>
                  Degradation-aware
                  <span className="block text-xs text-muted-foreground">
                    Prices battery wear from each battery&apos;s SOH and chemistry.
                  </span>
                </span>
              </label>
              <Field label="Replacement cost (per kWh)">
                <Input
                  type="number"
                  min={1}
                  value={replacementCost}
                  onChange={(e) => setReplacementCost(e.target.value)}
                  disabled={!degradation}
                />
              </Field>
              <Field label="End-of-horizon SOC">
                <Select
                  value={terminal}
                  onChange={(e) => setTerminal(e.target.value as "value" | "hold")}
                >
                  <option value="value">Value stored energy at mean price</option>
                  <option value="hold">Hold at least the initial SOC</option>
                </Select>
              </Field>
            </div>

            {formError && (
              <p role="alert" className="text-sm text-destructive">
                {formError}
              </p>
            )}

            <div className="flex flex-wrap gap-2">
              <Button type="submit" disabled={run.isPending} data-testid="schedule-run">
                <Play className="h-4 w-4" aria-hidden="true" />
                {run.isPending ? "Solving…" : "Optimise schedule"}
              </Button>
              {source === "series" && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() =>
                    setPricesText(sampleDayPrices((24 * 60) / intervalMin, intervalMin).join(", "))
                  }
                >
                  <Wand2 className="h-4 w-4" aria-hidden="true" />
                  Fill a sample day
                </Button>
              )}
            </div>
          </form>
        </CardContent>
      </Card>

      {run.isError && (
        <ErrorState
          title="Schedule optimisation failed."
          error={run.error}
          message={apiErrorMessage(run.error, "")}
        />
      )}
      {run.data && <ScheduleResult r={run.data} />}
    </div>
  );
}

function ScheduleResult({ r }: { r: ScheduleResponse }) {
  const names = useMemo(() => {
    const m = new Map<string, string>();
    for (const res of r.resources) {
      if (typeof res.id === "string") m.set(res.id, String(res.name ?? res.id));
    }
    return m;
  }, [r.resources]);
  const socSeries = Object.entries(r.per_resource).map(([id, plan]) => ({
    name: names.get(id) ?? id.slice(0, 8),
    soc: plan.soc,
  }));
  const throughput = r.charge.reduce((a, b) => a + b, 0) * (r.interval_minutes / 60);

  return (
    <Card data-testid="schedule-result">
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="text-base text-foreground">Result</CardTitle>
          <Badge variant={r.fallback_used ? "secondary" : "success"}>{r.status}</Badge>
          <Badge variant="outline" className="font-mono">
            {r.method}
          </Badge>
        </div>
        <Link
          href={`/trading/dispatches?run=${encodeURIComponent(r.run_id)}&view=explain`}
          className="inline-flex items-center gap-1 text-sm font-medium text-primary underline-offset-4 hover:underline"
          data-testid="explain-link"
        >
          Open in dispatch explainer
          <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
        </Link>
      </CardHeader>
      <CardContent className="space-y-4">
        {r.fallback_used && (
          <p role="status" className="rounded-md border border-amber-300 bg-amber-50 p-2 text-xs text-amber-900">
            The solver was not used; a rule-based fallback produced this schedule
            {r.fallback_reason ? `: ${r.fallback_reason}` : "."}
          </p>
        )}
        <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Energy cost" value={formatNumber(r.energy_cost)} testId="stat-energy-cost" />
          <Stat label="Wear cost" value={formatNumber(r.wear_cost)} />
          <Stat label="Objective" value={formatNumber(r.objective_value)} />
          <Stat label="Charged energy" value={`${formatNumber(throughput, 1)} kWh`} />
          <Stat label="Steps" value={`${r.power.length} × ${r.interval_minutes} min`} />
          <Stat label="Solve time" value={`${formatNumber(r.solve_time_ms, 0)} ms`} />
        </dl>
        <p className="text-xs text-muted-foreground">
          Negative cost = net revenue. Terminal SOC policy: {r.terminal_soc_policy}.
          {r.tariff_id ? ` Prices derived from tariff ${r.tariff_id}.` : ""}
        </p>
        <section aria-label="Power and price">
          <h3 className="mb-1 text-sm font-semibold">Fleet power vs. price</h3>
          <PowerPriceChart power={r.power} prices={r.prices} intervalMinutes={r.interval_minutes} />
        </section>
        <section aria-label="State of charge">
          <h3 className="mb-1 text-sm font-semibold">State of charge</h3>
          <SocChart series={socSeries} intervalMinutes={r.interval_minutes} />
        </section>
        {r.notes.length > 0 && (
          <ul className="list-disc space-y-0.5 pl-5 text-xs text-muted-foreground">
            {r.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        )}
        <p className="text-[11px] text-muted-foreground">
          Run <code>{r.run_id}</code>
        </p>
      </CardContent>
    </Card>
  );
}
