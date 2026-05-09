"use client";

import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { getHealth } from "@/lib/api/health";

export function HealthPill() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 10_000,
  });

  if (isLoading)
    return (
      <Badge variant="outline" aria-live="polite">
        checking…
      </Badge>
    );
  if (isError || !data || data.status !== "ok")
    return (
      <Badge variant="destructive" aria-live="polite">
        backend unreachable
      </Badge>
    );
  return (
    <Badge variant="success" aria-live="polite">
      backend ok
    </Badge>
  );
}
