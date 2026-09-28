"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

export const PowerPriceChart = dynamic(
  () => import("./charts").then((m) => m.PowerPriceChart),
  { ssr: false, loading: () => <Skeleton className="h-64 w-full" /> },
);

export const SocChart = dynamic(() => import("./charts").then((m) => m.SocChart), {
  ssr: false,
  loading: () => <Skeleton className="h-56 w-full" />,
});

export const CostBars = dynamic(() => import("./charts").then((m) => m.CostBars), {
  ssr: false,
  loading: () => <Skeleton className="h-48 w-full" />,
});
