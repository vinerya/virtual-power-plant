"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  apiErrorMessage,
  listTariffs,
  simulateBill,
  simulationToBill,
} from "@/lib/api/tariffs";
import type { BillSimulation, NemRegime, SimulateRequest, Tariff } from "@/lib/api/tariffs";
import { BillBreakdown } from "./bill-breakdown";
import { NEM_LABELS, money } from "./format";

const TIMEZONES = [
  "UTC",
  "America/Los_Angeles",
  "America/Denver",
  "America/Chicago",
  "America/New_York",
  "Europe/London",
  "Europe/Berlin",
  "Australia/Sydney",
];

function browserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

function firstOfThisMonth(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-01`;
}

const fieldLabel = "text-xs font-medium text-muted-foreground";
const selectCls =
  "mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm";

export function BillSimulator({ tariff }: { tariff: Tariff }) {
  const [source, setSource] = useState<"synthetic" | "csv">("synthetic");
  const [profile, setProfile] = useState<"" | "residential" | "commercial">("");
  const [avgKw, setAvgKw] = useState("");
  const [pvKw, setPvKw] = useState("0");
  const [startDate, setStartDate] = useState(firstOfThisMonth);
  const [days, setDays] = useState("30");
  const [csv, setCsv] = useState<string | null>(null);
  const [csvName, setCsvName] = useState<string | null>(null);
  const [timezone, setTimezone] = useState(browserTimezone);
  const [nem, setNem] = useState<"" | NemRegime>("");
  const [compareTo, setCompareTo] = useState<string>("");

  const tariffs = useQuery({
    queryKey: ["tariffs"],
    queryFn: listTariffs,
    staleTime: 60_000,
  });

  const zones = useMemo(
    () => (TIMEZONES.includes(timezone) ? TIMEZONES : [timezone, ...TIMEZONES]),
    [timezone],
  );

  const sim = useMutation({
    mutationFn: () => {
      const common: SimulateRequest = {
        timezone,
        nem: nem || undefined,
        compare_to: compareTo || undefined,
      };
      if (source === "csv") {
        return simulateBill(tariff.id, { ...common, csv: csv ?? "" });
      }
      const avg = parseFloat(avgKw);
      const pv = parseFloat(pvKw);
      return simulateBill(tariff.id, {
        ...common,
        synthetic: {
          profile: profile || undefined,
          avg_kw: Number.isFinite(avg) && avg > 0 ? avg : undefined,
          pv_kw: Number.isFinite(pv) && pv > 0 ? pv : 0,
        },
        period_days: Math.max(1, Math.min(366, parseInt(days, 10) || 30)),
        billing_period_start: startDate ? `${startDate}T00:00:00` : undefined,
      });
    },
    onError: (e: unknown) => toast.error(apiErrorMessage(e, "Simulation failed")),
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
    setSource("csv");
  };

  const error = sim.isError ? apiErrorMessage(sim.error, "Simulation failed") : null;

  return (
    <div className="space-y-4" data-testid="bill-simulator">
      <fieldset className="space-y-3 rounded-md border p-4">
        <legend className="px-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Load profile
        </legend>
        <div className="flex flex-wrap gap-4 text-sm" role="radiogroup" aria-label="Load source">
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="load-source"
              checked={source === "synthetic"}
              onChange={() => setSource("synthetic")}
              data-testid="synthetic-toggle"
            />
            Synthetic load (illustrative)
          </label>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="load-source"
              checked={source === "csv"}
              onChange={() => setSource("csv")}
              data-testid="csv-toggle"
            />
            Upload meter CSV
          </label>
        </div>

        {source === "synthetic" ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <label htmlFor="sim-profile" className={fieldLabel}>
                Shape
              </label>
              <select
                id="sim-profile"
                value={profile}
                onChange={(e) => setProfile(e.target.value as typeof profile)}
                className={selectCls}
              >
                <option value="">From tariff sector</option>
                <option value="residential">Residential</option>
                <option value="commercial">Commercial</option>
              </select>
            </div>
            <div>
              <label htmlFor="sim-avg-kw" className={fieldLabel}>
                Average load (kW)
              </label>
              <Input
                id="sim-avg-kw"
                inputMode="decimal"
                placeholder="default"
                value={avgKw}
                onChange={(e) => setAvgKw(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <label htmlFor="sim-pv-kw" className={fieldLabel}>
                Rooftop PV (kW peak)
              </label>
              <Input
                id="sim-pv-kw"
                inputMode="decimal"
                value={pvKw}
                onChange={(e) => setPvKw(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <label htmlFor="sim-start" className={fieldLabel}>
                Start date
              </label>
              <Input
                id="sim-start"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <label htmlFor="sim-days" className={fieldLabel}>
                Days
              </label>
              <Input
                id="sim-days"
                type="number"
                min={1}
                max={366}
                value={days}
                onChange={(e) => setDays(e.target.value)}
                className="mt-1"
              />
            </div>
          </div>
        ) : (
          <div>
            <label htmlFor="csv-upload" className={fieldLabel}>
              Meter trace CSV
            </label>
            <input
              id="csv-upload"
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              className="mt-1 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-3 file:py-1.5 file:text-sm hover:file:bg-muted/80"
              aria-describedby="csv-help"
              data-testid="csv-upload"
            />
            <p id="csv-help" className="mt-1 text-xs text-muted-foreground">
              Header row with <code>timestamp</code> (interval start, ISO 8601) and either{" "}
              <code>kw</code> (average kW; negative = export) or <code>import_kwh</code>
              [,<code>export_kwh</code>]. Times without an offset are read in the timezone
              below. {csvName ?? "No file chosen."}
            </p>
          </div>
        )}

        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <label htmlFor="sim-tz" className={fieldLabel}>
              Timezone (TOU hours)
            </label>
            <select
              id="sim-tz"
              value={timezone}
              onChange={(e) => setTimezone(e.target.value)}
              className={selectCls}
            >
              {zones.map((z) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="sim-nem" className={fieldLabel}>
              Export credit
            </label>
            <select
              id="sim-nem"
              value={nem}
              onChange={(e) => setNem(e.target.value as typeof nem)}
              className={selectCls}
            >
              <option value="">
                Tariff default ({NEM_LABELS[tariff.nem_regime] ?? tariff.nem_regime})
              </option>
              {Object.entries(NEM_LABELS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="compare-to" className={fieldLabel}>
              Compare to (optional)
            </label>
            <select
              id="compare-to"
              value={compareTo}
              onChange={(e) => setCompareTo(e.target.value)}
              className={selectCls}
            >
              <option value="">— none —</option>
              {(tariffs.data ?? [])
                .filter((t) => t.id !== tariff.id)
                .map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))}
            </select>
          </div>
        </div>

        <Button
          type="button"
          onClick={() => sim.mutate()}
          disabled={sim.isPending || (source === "csv" && !csv)}
          data-testid="run-simulation"
        >
          <Play className="mr-2 h-3.5 w-3.5" />
          {sim.isPending ? "Simulating…" : "Run simulation"}
        </Button>
        {error && (
          <p role="alert" className="text-sm text-destructive" data-testid="simulation-error">
            {error}
          </p>
        )}
      </fieldset>

      {sim.isPending && <Skeleton className="h-48 w-full" />}
      {sim.data && !sim.isPending && <SimulationResult sim={sim.data} />}
    </div>
  );
}

function SimulationResult({ sim }: { sim: BillSimulation }) {
  const bill = simulationToBill(sim);
  const cmp = sim.comparison ? simulationToBill(sim.comparison) : null;
  const ls = sim.load_summary;
  return (
    <div className="space-y-4" data-testid="simulation-result">
      {ls && (
        <dl
          className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-md border p-3 text-xs sm:grid-cols-4"
          data-testid="load-summary"
        >
          <Stat label="Load" value={ls.source === "synthetic" ? "Synthetic" : ls.source === "csv" ? "CSV upload" : "Meter trace"} />
          <Stat label="Imported" value={`${ls.import_kwh.toFixed(1)} kWh`} />
          <Stat label="Exported" value={`${ls.export_kwh.toFixed(1)} kWh`} />
          <Stat label="Peak" value={`${ls.peak_kw.toFixed(2)} kW`} />
          <Stat
            label="Period"
            value={`${sim.period_start.slice(0, 10)} → ${sim.period_end.slice(0, 10)}`}
          />
          <Stat label="Intervals" value={`${ls.intervals} × ${ls.interval_minutes} min`} />
          <Stat label="Timezone" value={ls.timezone} />
          <Stat
            label="Export credit"
            value={`${NEM_LABELS[sim.nem_regime] ?? sim.nem_regime}${
              sim.export_credit ? ` · ${money(-sim.export_credit)}` : ""
            }`}
          />
          {ls.method && (
            <p className="col-span-full pt-1 text-muted-foreground">{ls.method}</p>
          )}
        </dl>
      )}
      {sim.notes.length > 0 && (
        <ul className="list-disc pl-5 text-xs text-muted-foreground">
          {sim.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        <BillBreakdown bill={bill} title={sim.tariff_name || "Selected tariff"} comparison={cmp} />
        {cmp && sim.comparison && (
          <BillBreakdown bill={cmp} title={`Compared: ${sim.comparison.tariff_name}`} />
        )}
      </div>
      {sim.cycles.length > 1 && (
        <table className="w-full text-sm" data-testid="bill-cycles">
          <caption className="pb-1 text-left text-xs text-muted-foreground">
            Billed as {sim.cycles.length} monthly cycles (fixed and minimum charges apply per
            cycle).
          </caption>
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-1 font-medium">Cycle</th>
              <th className="py-1 text-right font-medium">Total</th>
            </tr>
          </thead>
          <tbody>
            {sim.cycles.map((c) => (
              <tr key={c.period_start} className="border-b last:border-b-0">
                <td className="py-1">
                  {c.period_start.slice(0, 10)} → {c.period_end.slice(0, 10)}
                </td>
                <td className="py-1 text-right tabular-nums">{money(c.total)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}
