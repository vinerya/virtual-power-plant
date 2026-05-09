// Local preset library for the tariff scaffolder. Used by the
// /api/tariff-presets route to allow operators to start from a canned
// shape (e.g. PG&E E-19, ConEd SC9) without round-tripping the backend.

import type { Tariff } from "@/lib/api/tariffs";

const PRESETS: Tariff[] = [
  {
    id: "pge-e-19",
    name: "PG&E E-19 (illustrative)",
    utility: "PG&E",
    sector: "commercial",
    components: [
      { name: "Customer charge", kind: "fixed", unit: "$/mo", rate: 50 },
      { name: "Energy — peak", kind: "energy", unit: "$/kWh", rate: 0.32 },
      { name: "Energy — partial peak", kind: "energy", unit: "$/kWh", rate: 0.18 },
      { name: "Energy — off peak", kind: "energy", unit: "$/kWh", rate: 0.10 },
      { name: "Demand — peak", kind: "demand", unit: "$/kW", rate: 18 },
    ],
  },
  {
    id: "coned-sc9",
    name: "ConEd SC9 (illustrative)",
    utility: "ConEd",
    sector: "commercial",
    components: [
      { name: "Customer charge", kind: "fixed", unit: "$/mo", rate: 28 },
      { name: "Energy — flat", kind: "energy", unit: "$/kWh", rate: 0.22 },
      { name: "Demand", kind: "demand", unit: "$/kW", rate: 22 },
    ],
  },
  {
    id: "flat-residential",
    name: "Flat residential",
    utility: "Generic",
    sector: "residential",
    components: [
      { name: "Customer charge", kind: "fixed", unit: "$/mo", rate: 12 },
      { name: "Energy", kind: "energy", unit: "$/kWh", rate: 0.15 },
    ],
  },
];

export interface TariffPresetSummary {
  id: string;
  name: string;
  utility?: string | null;
  sector?: string | null;
}

export function listPresetSummaries(): TariffPresetSummary[] {
  return PRESETS.map((t) => ({
    id: t.id,
    name: t.name,
    utility: t.utility,
    sector: t.sector,
  }));
}

export function getPreset(id: string): Tariff | null {
  return PRESETS.find((p) => p.id === id) ?? null;
}
