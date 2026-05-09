"use client";

import { Sun, Home, Zap } from "lucide-react";
import { cn, formatPower } from "@/lib/utils";

interface Flows {
  solar_to_home: number;
  solar_to_grid: number;
  grid_to_home: number;
  battery_to_home?: number;
  battery_charging?: number;
}

export function EnergyFlow({
  flows = {
    solar_to_home: 2.4,
    solar_to_grid: 1.1,
    grid_to_home: 0.6,
  },
}: {
  flows?: Flows;
}) {
  return (
    <div
      role="figure"
      aria-label="Energy flow: solar, home, grid"
      data-testid="energy-flow"
      className="rounded-md border bg-card p-4"
    >
      <h3 className="mb-3 text-sm font-semibold">Energy right now</h3>
      <div className="grid grid-cols-3 items-center gap-4 text-center">
        <Node icon={<Sun className="h-7 w-7" />} label="Solar" tone="amber" />
        <Node icon={<Home className="h-7 w-7" />} label="Home" tone="primary" />
        <Node icon={<Zap className="h-7 w-7" />} label="Grid" tone="muted" />
      </div>
      <ul className="mt-4 space-y-1.5 text-sm">
        <Flow
          label="Solar → Home"
          value={flows.solar_to_home}
          accent="emerald"
        />
        <Flow
          label="Solar → Grid"
          value={flows.solar_to_grid}
          accent="emerald"
        />
        <Flow
          label="Grid → Home"
          value={flows.grid_to_home}
          accent="amber"
        />
      </ul>
    </div>
  );
}

function Node({
  icon,
  label,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  tone: "amber" | "primary" | "muted";
}) {
  const cls =
    tone === "amber"
      ? "bg-amber-100 text-amber-700"
      : tone === "primary"
        ? "bg-primary/10 text-primary"
        : "bg-muted text-foreground";
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={cn("grid h-14 w-14 place-items-center rounded-full", cls)}
        aria-hidden="true"
      >
        {icon}
      </div>
      <span className="text-xs font-medium">{label}</span>
    </div>
  );
}

function Flow({
  label,
  value,
  accent,
}: {
  label: string;
  value: number;
  accent: "emerald" | "amber";
}) {
  return (
    <li className="flex items-center justify-between rounded border bg-background px-3 py-2">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={cn(
          "font-medium tabular-nums",
          accent === "emerald" ? "text-emerald-600" : "text-amber-600",
        )}
      >
        {formatPower(value)}
      </span>
    </li>
  );
}
