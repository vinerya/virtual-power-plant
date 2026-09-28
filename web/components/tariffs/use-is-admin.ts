"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api/client";

interface Me {
  username: string;
  role: string;
}

/**
 * Whether the signed-in user may create/edit/delete/import tariffs.
 * UX gating only — the API enforces the admin role on every write.
 */
export function useIsAdmin(): boolean {
  const me = useQuery({
    queryKey: ["auth", "me"],
    queryFn: () => api.get<Me>("/api/v1/auth/me"),
    staleTime: 5 * 60_000,
    retry: false,
  });
  return me.data?.role === "admin";
}
