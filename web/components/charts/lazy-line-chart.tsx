"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Recharts is heavy (~120KB gz). Lazy-load with SSR off so the asset
 * page's initial Lighthouse score isn't penalized.
 */
export const LineSeriesChart = dynamic(
  () => import("./line-series-chart").then((m) => m.LineSeriesChart),
  {
    ssr: false,
    loading: () => <Skeleton className="h-64 w-full" />,
  },
);

export const AreaScheduleChart = dynamic(
  () => import("./area-schedule-chart").then((m) => m.AreaScheduleChart),
  {
    ssr: false,
    loading: () => <Skeleton className="h-48 w-full" />,
  },
);
