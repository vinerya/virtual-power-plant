"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

export const PriceTrace = dynamic(() => import("./charts").then((m) => m.PriceTrace), {
  ssr: false,
  loading: () => <Skeleton className="h-40 w-full" />,
});

export const EquityCurve = dynamic(() => import("./charts").then((m) => m.EquityCurve), {
  ssr: false,
  loading: () => <Skeleton className="h-64 w-full" />,
});
