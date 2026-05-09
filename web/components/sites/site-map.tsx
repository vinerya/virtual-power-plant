"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Map, {
  Marker,
  NavigationControl,
  Popup,
  Source,
  Layer,
  type MapRef,
} from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Site } from "@/lib/api/types";
import { cn } from "@/lib/utils";
import { SitePopover } from "./site-popover";

// OpenFreeMap (https://openfreemap.org) ships free OSM-derived vector tiles
// without an API key. Documented in web/README.md.
const TILE_STYLE = "https://tiles.openfreemap.org/styles/positron";

interface LayerToggles {
  heatmap: boolean;
  capacity: boolean;
  region: boolean;
}

export function SiteMap({
  sites,
  selectedId,
  onSelect,
  flyToId,
  layers,
}: {
  sites: Site[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  flyToId: string | null;
  layers: LayerToggles;
}) {
  const mapRef = useRef<MapRef | null>(null);
  const [hover, setHover] = useState<Site | null>(null);

  // Fly to whenever the parent updates flyToId.
  useEffect(() => {
    if (!flyToId || !mapRef.current) return;
    const s = sites.find((x) => x.id === flyToId);
    if (!s) return;
    mapRef.current.flyTo({
      center: [s.lon, s.lat],
      zoom: 8,
      duration: 1200,
      essential: true,
    });
  }, [flyToId, sites]);

  const heatmapData = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: sites.map((s) => ({
        type: "Feature" as const,
        properties: {
          weight: Math.min(s.active_alerts + 1, 8),
        },
        geometry: { type: "Point" as const, coordinates: [s.lon, s.lat] },
      })),
    }),
    [sites],
  );

  const initialView = useMemo(() => {
    if (sites.length === 0) return { longitude: -98, latitude: 39, zoom: 3.5 };
    const lon = sites.reduce((s, x) => s + x.lon, 0) / sites.length;
    const lat = sites.reduce((s, x) => s + x.lat, 0) / sites.length;
    return { longitude: lon, latitude: lat, zoom: 3.8 };
  }, [sites]);

  return (
    <div className="relative h-full w-full" data-testid="site-map">
      <Map
        ref={(r) => {
          mapRef.current = r;
        }}
        initialViewState={initialView}
        mapStyle={TILE_STYLE}
        attributionControl={true}
      >
        <NavigationControl position="top-right" />

        {layers.heatmap && (
          <Source id="alert-heat" type="geojson" data={heatmapData}>
            <Layer
              id="alert-heat-layer"
              type="heatmap"
              paint={{
                "heatmap-weight": ["get", "weight"],
                "heatmap-intensity": 1,
                "heatmap-radius": 30,
                "heatmap-opacity": 0.55,
                "heatmap-color": [
                  "interpolate",
                  ["linear"],
                  ["heatmap-density"],
                  0,
                  "rgba(33,102,172,0)",
                  0.3,
                  "rgba(255,200,100,0.6)",
                  0.7,
                  "rgba(239,68,68,0.7)",
                  1,
                  "rgba(127,29,29,0.85)",
                ],
              }}
            />
          </Source>
        )}

        {sites.map((s) => (
          <Marker
            key={s.id}
            longitude={s.lon}
            latitude={s.lat}
            anchor="center"
          >
            <button
              type="button"
              data-testid="site-marker"
              data-site-id={s.id}
              aria-label={`Site ${s.name}`}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(s.id);
              }}
              onMouseEnter={() => setHover(s)}
              onMouseLeave={() => setHover((cur) => (cur?.id === s.id ? null : cur))}
              className={cn(
                "group relative grid place-items-center rounded-full border border-white shadow",
                healthBg(s.health),
                selectedId === s.id ? "ring-2 ring-primary" : "",
              )}
              style={
                layers.capacity
                  ? {
                      width: bubbleSize(s.rated_power),
                      height: bubbleSize(s.rated_power),
                    }
                  : { width: 14, height: 14 }
              }
            />
          </Marker>
        ))}

        {hover && (
          <Popup
            longitude={hover.lon}
            latitude={hover.lat}
            closeButton={false}
            closeOnClick={false}
            anchor="top"
            offset={12}
          >
            <SitePopover site={hover} />
          </Popup>
        )}
      </Map>
    </div>
  );
}

function healthBg(h: Site["health"]): string {
  if (h === "red") return "bg-destructive";
  if (h === "yellow") return "bg-amber-500";
  return "bg-emerald-500";
}

function bubbleSize(ratedKw: number): number {
  // Scale rated capacity (kW) to a 12–48px bubble.
  const v = Math.max(0, Math.min(1, Math.log10(Math.max(1, ratedKw)) / 4));
  return 12 + v * 36;
}
