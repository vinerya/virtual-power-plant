"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { FlaskConical, Play } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Field, Stat, pnlTone } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  backtestStrategy,
  listStrategies,
  type Strategy,
  type StrategyBacktest,
} from "@/lib/api/trading";
import { apiErrorMessage } from "@/lib/api/errors";
import { formatMoney, formatNumber, formatPercent, parseApiDate } from "@/lib/utils";
import { EquityCurve } from "./lazy-charts";

type ParamDraft = Record<string, string>;

function toDraft(params: Record<string, unknown>): ParamDraft {
  const out: ParamDraft = {};
  for (const [k, v] of Object.entries(params)) {
    out[k] = JSON.stringify(v);
  }
  return out;
}

function fromDraft(draft: ParamDraft): { params: Record<string, unknown>; errors: Record<string, string> } {
  const params: Record<string, unknown> = {};
  const errors: Record<string, string> = {};
  for (const [k, raw] of Object.entries(draft)) {
    try {
      params[k] = JSON.parse(raw);
    } catch {
      errors[k] = "Not valid JSON (numbers as-is, text in quotes, lists in [ ])";
    }
  }
  return { params, errors };
}

export function StrategiesView() {
  const strategies = useQuery({ queryKey: ["trading", "strategies"], queryFn: listStrategies });
  const [name, setName] = useState<string>("");
  const current = useMemo(
    () => strategies.data?.find((s) => s.name === name),
    [strategies.data, name],
  );

  useEffect(() => {
    if (!name && strategies.data?.length) setName(strategies.data[0].name);
  }, [strategies.data, name]);

  if (strategies.isLoading) return <Skeleton className="h-64 w-full" />;
  if (strategies.error || !strategies.data)
    return (
      <ErrorState
        title="Failed to load strategies."
        error={strategies.error}
        onRetry={() => void strategies.refetch()}
      />
    );

  return (
    <div className="space-y-4" data-testid="strategies-view">
      <ul className="grid gap-3 md:grid-cols-2" aria-label="Available strategies">
        {strategies.data.map((s) => (
          <li key={s.name}>
            <Card className={s.name === name ? "border-primary" : undefined}>
              <CardHeader className="flex-row items-start justify-between gap-2 space-y-0 pb-2">
                <div>
                  <CardTitle className="text-base text-foreground">{s.name}</CardTitle>
                  <p className="mt-1 text-xs text-muted-foreground">{s.description}</p>
                </div>
                <Button
                  type="button"
                  size="sm"
                  variant={s.name === name ? "default" : "outline"}
                  onClick={() => setName(s.name)}
                  aria-pressed={s.name === name}
                  aria-label={`Backtest ${s.name}`}
                >
                  Backtest
                </Button>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="flex flex-wrap gap-1">
                  {s.min_markets > 1 && (
                    <Badge variant="outline">needs {s.min_markets} markets</Badge>
                  )}
                  {Object.entries(s.parameters).map(([k, v]) => (
                    <Badge key={k} variant="secondary" className="font-mono font-normal">
                      {k}={JSON.stringify(v)}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </li>
        ))}
      </ul>

      {current && <BacktestPanel key={current.name} strategy={current} />}
    </div>
  );
}

function BacktestPanel({ strategy }: { strategy: Strategy }) {
  const [draft, setDraft] = useState<ParamDraft>(() => toDraft(strategy.parameters));
  const [periods, setPeriods] = useState("168");
  const [intervalMin, setIntervalMin] = useState("60");
  const [seed, setSeed] = useState("42");
  const [cash, setCash] = useState("100000");
  const [fee, setFee] = useState("0.07");
  const [halfSpread, setHalfSpread] = useState("0.001");
  const [paramErrors, setParamErrors] = useState<Record<string, string>>({});

  const run = useMutation({
    mutationFn: (body: Parameters<typeof backtestStrategy>[1]) =>
      backtestStrategy(strategy.name, body),
  });

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const { params, errors } = fromDraft(draft);
    setParamErrors(errors);
    if (Object.keys(errors).length) return;
    run.mutate({
      params,
      synthetic: {
        periods: Number(periods),
        interval_minutes: Number(intervalMin),
        seed: Number(seed),
      },
      initial_cash: Number(cash),
      fee_per_unit: Number(fee),
      half_spread: Number(halfSpread),
    });
  }

  return (
    <Card data-testid="backtest-panel">
      <CardHeader className="pb-3">
        <CardTitle className="text-base text-foreground">Backtest: {strategy.name}</CardTitle>
        <p className="text-xs text-muted-foreground">
          Runs over seeded synthetic prices for the simulated markets. Results are indicative
          only; see the assumptions listed with each result.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={onSubmit} className="space-y-4" aria-label={`Backtest ${strategy.name}`}>
          <fieldset className="space-y-2">
            <legend className="text-sm font-semibold">Strategy parameters</legend>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {Object.keys(draft).map((k) => (
                <Field key={k} label={k} error={paramErrors[k]}>
                  <Input
                    value={draft[k]}
                    onChange={(e) => setDraft((d) => ({ ...d, [k]: e.target.value }))}
                    className="font-mono"
                    data-testid={`param-${k}`}
                  />
                </Field>
              ))}
            </div>
          </fieldset>
          <fieldset className="space-y-2">
            <legend className="text-sm font-semibold">Data &amp; costs</legend>
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <Field label="Bars" hint="2 – 20,000">
                <Input
                  type="number"
                  min={2}
                  max={20000}
                  value={periods}
                  onChange={(e) => setPeriods(e.target.value)}
                  data-testid="bt-periods"
                />
              </Field>
              <Field label="Bar length (min)">
                <Select value={intervalMin} onChange={(e) => setIntervalMin(e.target.value)}>
                  {[5, 15, 30, 60, 240, 1440].map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Seed">
                <Input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
              </Field>
              <Field label="Initial cash ($)">
                <Input
                  type="number"
                  min={1}
                  value={cash}
                  onChange={(e) => setCash(e.target.value)}
                />
              </Field>
              <Field label="Fee per MWh ($)">
                <Input
                  type="number"
                  min={0}
                  step={0.01}
                  value={fee}
                  onChange={(e) => setFee(e.target.value)}
                />
              </Field>
              <Field label="Half spread" hint="fraction of price">
                <Input
                  type="number"
                  min={0}
                  max={0.49}
                  step={0.001}
                  value={halfSpread}
                  onChange={(e) => setHalfSpread(e.target.value)}
                />
              </Field>
            </div>
          </fieldset>
          <Button type="submit" disabled={run.isPending} data-testid="bt-run">
            <Play className="h-4 w-4" aria-hidden="true" />
            {run.isPending ? "Running…" : "Run backtest"}
          </Button>
        </form>

        {run.isError && (
          <ErrorState
            title="Backtest failed."
            error={run.error}
            message={apiErrorMessage(run.error, "")}
          />
        )}
        {run.data && <BacktestResult r={run.data} />}
      </CardContent>
    </Card>
  );
}

function BacktestResult({ r }: { r: StrategyBacktest }) {
  const curve = useMemo(
    () =>
      r.equity_curve
        .map((p) => ({ t: parseApiDate(p.timestamp)?.getTime() ?? NaN, equity: p.equity }))
        .filter((p) => Number.isFinite(p.t)),
    [r.equity_curve],
  );
  return (
    <section aria-label="Backtest result" className="space-y-3" data-testid="bt-result">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline" className="gap-1">
          <FlaskConical className="h-3 w-3" aria-hidden="true" />
          {r.data_source === "synthetic" ? "synthetic prices" : "provided prices"}
        </Badge>
        {r.periods} bars × {r.interval_minutes} min on {r.markets.join(", ")}
      </div>
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat
          label="Total P&L"
          value={formatMoney(r.total_pnl)}
          tone={pnlTone(r.total_pnl)}
          sub={`return ${formatPercent(r.total_return)}`}
          testId="bt-total-pnl"
        />
        <Stat label="Final equity" value={formatMoney(r.final_equity)} />
        <Stat label="Sharpe (annualised)" value={formatNumber(r.sharpe_ratio)} />
        <Stat label="Max drawdown" value={formatPercent(r.max_drawdown)} />
        <Stat
          label="Realized / unrealized"
          value={formatMoney(r.realized_pnl)}
          sub={`unrealized ${formatMoney(r.unrealized_pnl)}`}
        />
        <Stat label="Fees" value={formatMoney(r.fees)} />
        <Stat label="Trades" value={String(r.num_trades)} />
        <Stat
          label="Win rate"
          value={r.win_rate != null ? formatPercent(r.win_rate) : "—"}
          sub={r.win_rate == null ? "no closing trades" : undefined}
        />
      </dl>
      {curve.length >= 2 && <EquityCurve data={curve} initial={r.initial_cash} />}
      {Object.keys(r.final_positions).length > 0 && (
        <p className="text-xs text-muted-foreground">
          Final positions:{" "}
          {Object.entries(r.final_positions)
            .map(([m, q]) => `${m} ${formatNumber(q, 1)}`)
            .join(", ")}
        </p>
      )}
      {r.assumptions.length > 0 && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            Assumptions ({r.assumptions.length})
          </summary>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {r.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
