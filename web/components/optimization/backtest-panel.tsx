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
  MAX_BACKTEST_STEPS,
  parseSeries,
  runBacktest,
  sampleDayPrices,
  type BacktestRequest,
  type BacktestResponse,
} from "@/lib/api/optimization";
import { cn, formatNumber, formatPercent } from "@/lib/utils";
import { CostBars, PowerPriceChart, SocChart } from "./lazy-charts";

const MAX_WORK = 672 * 48;

export function BacktestPanel() {
  const resources = useQuery({ queryKey: ["resources"], queryFn: listResources, staleTime: 60_000 });
  const batteries = useMemo(
    () => (resources.data ?? []).filter((r) => r.resource_type === "battery" && r.online),
    [resources.data],
  );

  const [batterySource, setBatterySource] = useState<"inline" | "resource">("inline");
  const [resourceId, setResourceId] = useState("");
  const [capacity, setCapacity] = useState("100");
  const [maxPower, setMaxPower] = useState("50");
  const [socInit, setSocInit] = useState("0.5");
  const [intervalMin, setIntervalMin] = useState(60);
  const [pricesText, setPricesText] = useState(() => sampleDayPrices(48, 60).join(", "));
  const [horizon, setHorizon] = useState("24");
  const [forecast, setForecast] = useState<BacktestRequest["forecast_mode"]>("persistence");
  const [noise, setNoise] = useState("0.1");
  const [seed, setSeed] = useState("42");
  const [compareOffline, setCompareOffline] = useState(true);
  const [formError, setFormError] = useState<string | null>(null);

  const parsed = useMemo(() => parseSeries(pricesText), [pricesText]);
  const run = useMutation({ mutationFn: runBacktest });

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);
    const n = parsed.values.length;
    const h = Number(horizon);
    if (parsed.invalid.length) return setFormError(`Not a number: ${parsed.invalid.slice(0, 3).join(", ")}`);
    if (n < 2 || n > MAX_BACKTEST_STEPS)
      return setFormError(`Enter between 2 and ${MAX_BACKTEST_STEPS} prices.`);
    if (!(h >= 2 && h <= 96)) return setFormError("Lookahead must be 2–96 steps.");
    if (n * h > MAX_WORK)
      return setFormError(`prices × lookahead must be ≤ ${MAX_WORK}; shorten one of them.`);
    const body: BacktestRequest = {
      prices: parsed.values,
      interval_minutes: intervalMin,
      horizon_steps: h,
      forecast_mode: forecast,
      compare_offline: compareOffline,
    };
    if (forecast === "noisy") {
      body.noise_sigma = Number(noise);
      body.seed = Number(seed);
    }
    if (batterySource === "resource") {
      if (!resourceId) return setFormError("Choose a battery resource.");
      body.resource_id = resourceId;
    } else {
      const cap = Number(capacity);
      const p = Number(maxPower);
      const soc = Number(socInit);
      if (!(cap > 0) || !(p > 0)) return setFormError("Capacity and power must be positive.");
      if (!(soc >= 0.05 && soc <= 0.95)) return setFormError("Initial SOC must be 0.05–0.95.");
      body.battery = { capacity_kwh: cap, max_power_kw: p, soc_init: soc };
    }
    run.mutate(body);
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base text-foreground">Closed-loop backtest</CardTitle>
          <p className="text-xs text-muted-foreground">
            Replays the MPC controller tick by tick over a historical price series: each tick
            it sees a forecast, commits only its first step, and the battery is simulated
            against the true prices. Compared with idling, the rule-based dispatcher and the
            perfect-foresight optimum.
          </p>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4" aria-label="Backtest request">
            <fieldset className="space-y-2">
              <legend className="text-xs font-medium text-muted-foreground">Battery</legend>
              <div className="flex flex-wrap gap-4 text-sm">
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="battery-source"
                    checked={batterySource === "inline"}
                    onChange={() => setBatterySource("inline")}
                  />
                  Hypothetical battery
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="battery-source"
                    checked={batterySource === "resource"}
                    onChange={() => setBatterySource("resource")}
                    disabled={batteries.length === 0}
                  />
                  Fleet battery{batteries.length === 0 ? " (none online)" : ""}
                </label>
              </div>
              {batterySource === "resource" ? (
                <Field label="Battery resource" hint="Uses its persisted capacity, SOC and SOH.">
                  <Select value={resourceId} onChange={(e) => setResourceId(e.target.value)}>
                    <option value="">Choose…</option>
                    {batteries.map((b) => (
                      <option key={b.id} value={b.id}>
                        {b.name}
                      </option>
                    ))}
                  </Select>
                </Field>
              ) : (
                <div className="grid gap-3 sm:grid-cols-3">
                  <Field label="Capacity (kWh)">
                    <Input
                      type="number"
                      min={0}
                      value={capacity}
                      onChange={(e) => setCapacity(e.target.value)}
                      data-testid="bt-capacity"
                    />
                  </Field>
                  <Field label="Max power (kW)">
                    <Input
                      type="number"
                      min={0}
                      value={maxPower}
                      onChange={(e) => setMaxPower(e.target.value)}
                    />
                  </Field>
                  <Field label="Initial SOC (0–1)">
                    <Input
                      type="number"
                      min={0.05}
                      max={0.95}
                      step={0.05}
                      value={socInit}
                      onChange={(e) => setSocInit(e.target.value)}
                    />
                  </Field>
                </div>
              )}
            </fieldset>

            <Field
              label="Historical prices (currency per kWh)"
              hint={`${parsed.values.length} steps = ${formatNumber((parsed.values.length * intervalMin) / 60, 1)} h`}
            >
              <textarea
                value={pricesText}
                onChange={(e) => setPricesText(e.target.value)}
                rows={3}
                className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                data-testid="bt-prices"
              />
            </Field>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <Field label="Interval (minutes)">
                <Select value={intervalMin} onChange={(e) => setIntervalMin(Number(e.target.value))}>
                  {INTERVALS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Lookahead (steps)">
                <Input
                  type="number"
                  min={2}
                  max={96}
                  value={horizon}
                  onChange={(e) => setHorizon(e.target.value)}
                />
              </Field>
              <Field
                label="Forecast"
                hint={
                  forecast === "perfect"
                    ? "Sees true future prices"
                    : forecast === "persistence"
                      ? "Day-ahead persistence"
                      : "True prices + seeded noise"
                }
              >
                <Select
                  value={forecast}
                  onChange={(e) => setForecast(e.target.value as BacktestRequest["forecast_mode"])}
                  data-testid="bt-forecast"
                >
                  <option value="perfect">Perfect</option>
                  <option value="persistence">Persistence</option>
                  <option value="noisy">Noisy</option>
                </Select>
              </Field>
              {forecast === "noisy" && (
                <>
                  <Field label="Noise σ">
                    <Input
                      type="number"
                      min={0}
                      max={2}
                      step={0.05}
                      value={noise}
                      onChange={(e) => setNoise(e.target.value)}
                    />
                  </Field>
                  <Field label="Seed">
                    <Input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
                  </Field>
                </>
              )}
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={compareOffline}
                onChange={(e) => setCompareOffline(e.target.checked)}
              />
              Also solve the perfect-foresight optimum (needed for regret)
            </label>

            {formError && (
              <p role="alert" className="text-sm text-destructive">
                {formError}
              </p>
            )}
            <div className="flex flex-wrap gap-2">
              <Button type="submit" disabled={run.isPending} data-testid="bt-submit">
                <Play className="h-4 w-4" aria-hidden="true" />
                {run.isPending ? "Running…" : "Run backtest"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() =>
                  setPricesText(sampleDayPrices((48 * 60) / intervalMin, intervalMin).join(", "))
                }
              >
                <Wand2 className="h-4 w-4" aria-hidden="true" />
                Fill two sample days
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {run.isError && (
        <ErrorState
          title="Backtest failed."
          error={run.error}
          message={apiErrorMessage(run.error, "")}
        />
      )}
      {run.data && <BacktestResult r={run.data} prices={run.variables?.prices ?? []} />}
    </div>
  );
}

function BacktestResult({ r, prices }: { r: BacktestResponse; prices: number[] }) {
  const rows = [
    { name: "MPC (this run)", cost: r.realized_cost_adjusted, highlight: true },
    { name: "Rule-based", cost: r.rules_cost_adjusted },
    { name: "Idle", cost: r.no_action_cost },
    ...(r.perfect_foresight_cost_adjusted != null
      ? [{ name: "Perfect foresight", cost: r.perfect_foresight_cost_adjusted }]
      : []),
  ];
  const vsRules = r.rules_cost_adjusted - r.realized_cost_adjusted;
  const vsIdle = r.no_action_cost - r.realized_cost_adjusted;
  // Share of the perfect-foresight value (vs idle) the controller captured.
  const pfValue =
    r.perfect_foresight_cost_adjusted != null
      ? r.no_action_cost - r.perfect_foresight_cost_adjusted
      : null;
  const captured = pfValue && Math.abs(pfValue) > 1e-9 ? vsIdle / pfValue : null;

  return (
    <Card data-testid="backtest-result">
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="text-base text-foreground">Backtest result</CardTitle>
          <Badge variant={r.fallback_count > 0 ? "secondary" : "success"}>{r.status}</Badge>
          <Badge variant="outline">{r.forecast_mode} forecast</Badge>
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
        <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <Stat
            label="Regret vs. perfect foresight"
            value={r.regret != null ? formatNumber(r.regret) : "—"}
            sub={
              r.regret == null
                ? `offline optimum: ${r.perfect_foresight_status}`
                : "cost above the best possible (lower is better)"
            }
            tone={r.regret != null && r.regret > 1e-6 ? "warning" : undefined}
            testId="stat-regret"
          />
          <Stat
            label="Saving vs. rule-based"
            value={formatNumber(vsRules)}
            tone={vsRules > 0 ? "positive" : vsRules < 0 ? "negative" : undefined}
          />
          <Stat
            label="Saving vs. idle"
            value={formatNumber(vsIdle)}
            tone={vsIdle > 0 ? "positive" : vsIdle < 0 ? "negative" : undefined}
          />
          <Stat
            label="Value captured"
            value={captured != null ? formatPercent(captured) : "—"}
            sub="of the perfect-foresight value"
          />
          <Stat label="Ticks / fallbacks" value={`${r.ticks} / ${r.fallback_count}`} />
          <Stat
            label="Solver time"
            value={`${formatNumber(r.cumulative_solve_time_ms, 0)} ms`}
            sub={`${r.cumulative_solver_iterations} iterations`}
          />
        </dl>

        <div className="grid gap-4 lg:grid-cols-2">
          <section aria-label="Cost comparison">
            <h3 className="mb-1 text-sm font-semibold">Total cost by policy (lower is better)</h3>
            <CostBars rows={rows} />
            <table className="mt-2 w-full text-xs" data-testid="cost-table">
              <caption className="sr-only">Cost by policy</caption>
              <thead className="text-muted-foreground">
                <tr>
                  <th scope="col" className="py-1 text-left font-medium">Policy</th>
                  <th scope="col" className="py-1 text-right font-medium">Cost (adjusted)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.name} className={cn("border-t", row.highlight && "font-semibold")}>
                    <th scope="row" className="py-1 text-left font-normal">
                      {row.name}
                    </th>
                    <td className="py-1 text-right tabular-nums">{formatNumber(row.cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-1 text-[11px] text-muted-foreground">
              &quot;Adjusted&quot; credits energy left in the battery at{" "}
              {formatNumber(r.terminal_energy_value_per_kwh, 4)} per kWh so policies that end
              with different SOC are comparable. Raw MPC cost {formatNumber(r.realized_cost)}.
            </p>
          </section>
          <section aria-label="Committed schedule">
            <h3 className="mb-1 text-sm font-semibold">Committed power vs. price</h3>
            <PowerPriceChart
              power={r.power}
              prices={prices.slice(0, r.power.length)}
              intervalMinutes={r.interval_minutes}
            />
          </section>
        </div>
        <section aria-label="State of charge">
          <h3 className="mb-1 text-sm font-semibold">
            State of charge (final {formatPercent(r.final_soc)})
          </h3>
          <SocChart series={[{ name: "Battery", soc: r.soc }]} intervalMinutes={r.interval_minutes} />
        </section>
        {r.notes.length > 0 && (
          <ul className="list-disc space-y-0.5 pl-5 text-xs text-muted-foreground">
            {r.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
