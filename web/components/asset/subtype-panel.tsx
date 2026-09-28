"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ResourceResponse } from "@/lib/api/types";
import { formatPercent, formatPower } from "@/lib/utils";

export function SubtypePanel({ r }: { r: ResourceResponse }) {
  switch (r.resource_type) {
    case "battery":
      return <BatteryPanel r={r} />;
    case "solar":
      return <SolarPanel r={r} />;
    case "wind_turbine":
    case "wind": // legacy alias
      return <WindPanel r={r} />;
    default:
      return <DefaultPanel r={r} />;
  }
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function BatteryPanel({ r }: { r: ResourceResponse }) {
  const soc = num(r.state_of_charge);
  const cap = num(r.capacity_kwh);
  const cycles = num(r.equivalent_full_cycles);
  // Unset limits mean "rated power" on the backend.
  const chargeLimit = num(r.max_charge_kw) ?? num(r.rated_power);
  const dischargeLimit = num(r.max_discharge_kw) ?? num(r.rated_power);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Battery</CardTitle>
      </CardHeader>
      <CardContent
        className="grid gap-6 md:grid-cols-[160px_1fr]"
        data-testid="subtype-battery"
      >
        <SocGauge value={soc ?? 0} known={soc != null} />
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
          <KV label="Capacity" value={cap != null ? `${cap.toFixed(1)} kWh` : "—"} />
          <KV
            label="Equivalent full cycles"
            value={cycles != null ? cycles.toFixed(1) : "—"}
          />
          <KV
            label="Charge limit"
            value={chargeLimit != null ? formatPower(chargeLimit) : "—"}
          />
          <KV
            label="Discharge limit"
            value={dischargeLimit != null ? formatPower(dischargeLimit) : "—"}
          />
        </dl>
      </CardContent>
    </Card>
  );
}

function SocGauge({ value, known }: { value: number; known: boolean }) {
  // Radial gauge built from two SVG circles. No extra deps.
  const pct = Math.max(0, Math.min(1, value));
  const r = 56;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - pct);
  return (
    <div
      className="flex flex-col items-center justify-center"
      role="img"
      aria-label={`State of charge ${known ? `${(pct * 100).toFixed(0)}%` : "unknown"}`}
    >
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle
          cx="70"
          cy="70"
          r={r}
          stroke="hsl(var(--muted))"
          strokeWidth="12"
          fill="none"
        />
        <circle
          cx="70"
          cy="70"
          r={r}
          stroke="hsl(var(--primary))"
          strokeWidth="12"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          transform="rotate(-90 70 70)"
        />
        <text
          x="70"
          y="76"
          textAnchor="middle"
          className="fill-foreground text-2xl font-semibold"
        >
          {known ? `${Math.round(pct * 100)}%` : "—"}
        </text>
      </svg>
      <span className="text-xs text-muted-foreground">SOC</span>
    </div>
  );
}

function SolarPanel({ r }: { r: ResourceResponse }) {
  const irr = num(r.irradiance);
  const dc = num(r.dc_capacity_kw);
  const ac = num(r.ac_capacity_kw);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Solar array</CardTitle>
      </CardHeader>
      <CardContent data-testid="subtype-solar">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm md:grid-cols-3">
          <KV
            label="Irradiance"
            value={irr != null ? `${irr.toFixed(0)} W/m²` : "—"}
          />
          <KV label="DC capacity" value={dc != null ? formatPower(dc) : "—"} />
          <KV label="AC capacity" value={ac != null ? formatPower(ac) : "—"} />
          <KV label="Current PV output" value={formatPower(r.current_power)} />
          <KV
            label="Efficiency"
            value={r.efficiency != null ? formatPercent(r.efficiency) : "—"}
          />
        </dl>
      </CardContent>
    </Card>
  );
}

function WindPanel({ r }: { r: ResourceResponse }) {
  const ws = num(r.wind_speed_ms);
  const cin = num(r.cut_in_speed_ms);
  const cout = num(r.cut_out_speed_ms);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Wind turbine</CardTitle>
      </CardHeader>
      <CardContent data-testid="subtype-wind">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm md:grid-cols-3">
          <KV
            label="Wind speed"
            value={ws != null ? `${ws.toFixed(1)} m/s` : "—"}
          />
          <KV
            label="Cut-in"
            value={cin != null ? `${cin.toFixed(1)} m/s` : "—"}
          />
          <KV
            label="Cut-out"
            value={cout != null ? `${cout.toFixed(1)} m/s` : "—"}
          />
          <KV label="Current power" value={formatPower(r.current_power)} />
          <KV label="Rated" value={formatPower(r.rated_power)} />
        </dl>
      </CardContent>
    </Card>
  );
}

function DefaultPanel({ r }: { r: ResourceResponse }) {
  const SKIP = new Set([
    "id",
    "name",
    "resource_type",
    "rated_power",
    "online",
    "current_power",
    "efficiency",
    "created_at",
    "updated_at",
    "metadata",
  ]);
  const rows = Object.entries(r)
    .filter(
      ([k, v]) => !SKIP.has(k) && (typeof v === "string" || typeof v === "number" || typeof v === "boolean"),
    )
    .sort(([a], [b]) => a.localeCompare(b));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Metrics</CardTitle>
      </CardHeader>
      <CardContent data-testid="subtype-default">
        {rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No subtype metrics reported.
          </p>
        ) : (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm md:grid-cols-3">
            {rows.map(([k, v]) => (
              <KV key={k} label={k.replace(/_/g, " ")} value={String(v)} />
            ))}
          </dl>
        )}
      </CardContent>
    </Card>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}
