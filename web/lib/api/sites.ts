import { api } from "./client";
import { listResources } from "./resources";
import { listAlerts } from "./alerts";
import type { ResourceResponse, Site } from "./types";

/**
 * Sites feed.
 *
 * Tries `/api/v1/sites` first. If the backend hasn't shipped that endpoint
 * yet (404), we synthesize sites from the resources list:
 *   1. Group resources by `metadata.site_id` if present.
 *   2. Use `metadata.location.{lat,lon}` when available.
 *   3. Fall back to a deterministic synthetic grid covering the
 *      continental US for resources without coordinates. This is purely
 *      for demo and is documented in the README.
 */
export async function listSites(): Promise<Site[]> {
  try {
    return await api.get<Site[]>("/api/v1/sites");
  } catch (e) {
    if ((e as { status?: number })?.status !== 404) throw e;
  }

  const [resources, alerts] = await Promise.all([
    listResources().catch(() => [] as ResourceResponse[]),
    listAlerts({ status: "active" }).catch(() => []),
  ]);

  const alertCountBySource = new Map<string, number>();
  for (const a of alerts) {
    if (a.status !== "active") continue;
    alertCountBySource.set(a.source, (alertCountBySource.get(a.source) ?? 0) + 1);
  }

  // Group by site_id (or singleton site = resource id).
  const groups = new Map<string, { name: string; resources: ResourceResponse[] }>();
  for (const r of resources) {
    const meta = (r.metadata ?? {}) as Record<string, unknown>;
    const siteId = (meta.site_id as string | undefined) ?? `site-${r.id}`;
    const siteName =
      (meta.site_name as string | undefined) ??
      (meta.site_id ? String(meta.site_id) : r.name);
    const g = groups.get(siteId) ?? { name: siteName, resources: [] };
    g.resources.push(r);
    groups.set(siteId, g);
  }

  const sites: Site[] = [];
  let i = 0;
  for (const [id, g] of groups) {
    const coords = pickCoords(g.resources, i++);
    const total = g.resources.length;
    const online = g.resources.filter((r) => r.online).length;
    const current = g.resources.reduce((s, r) => s + (r.current_power || 0), 0);
    const rated = g.resources.reduce((s, r) => s + (r.rated_power || 0), 0);
    const sohLow = g.resources.some(
      (r) => typeof r.state_of_health === "number" && r.state_of_health < 0.85,
    );
    const offline = total - online;
    const activeAlerts = g.resources.reduce(
      (s, r) => s + (alertCountBySource.get(r.id) ?? 0),
      0,
    );
    const health = scoreHealth({ activeAlerts, sohLow, offline });
    sites.push({
      id,
      name: g.name,
      lat: coords.lat,
      lon: coords.lon,
      region: coords.region,
      resource_ids: g.resources.map((r) => r.id),
      total_resources: total,
      online_count: online,
      current_power: current,
      rated_power: rated,
      active_alerts: activeAlerts,
      health,
    });
  }
  return sites;
}

function pickCoords(
  resources: ResourceResponse[],
  fallbackIndex: number,
): { lat: number; lon: number; region?: string } {
  for (const r of resources) {
    const meta = (r.metadata ?? {}) as Record<string, unknown>;
    const loc = meta.location as
      | { lat?: number; lon?: number; region?: string }
      | undefined;
    if (
      loc &&
      typeof loc.lat === "number" &&
      typeof loc.lon === "number" &&
      Number.isFinite(loc.lat) &&
      Number.isFinite(loc.lon)
    ) {
      return { lat: loc.lat, lon: loc.lon, region: loc.region };
    }
  }
  // Deterministic synthetic grid across continental US.
  const cols = 8;
  const row = Math.floor(fallbackIndex / cols);
  const col = fallbackIndex % cols;
  const lat = 32 + row * 2.5;
  const lon = -118 + col * 6;
  return { lat, lon, region: "demo-grid" };
}

function scoreHealth(args: {
  activeAlerts: number;
  sohLow: boolean;
  offline: number;
}): "green" | "yellow" | "red" {
  const { activeAlerts, sohLow, offline } = args;
  if (activeAlerts >= 3 || offline >= 2) return "red";
  if (activeAlerts >= 1 || sohLow || offline >= 1) return "yellow";
  return "green";
}
