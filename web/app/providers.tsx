"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { Toaster } from "@/components/ui/sonner";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 2_000,
            refetchOnWindowFocus: false,
            // Client errors (4xx) won't fix themselves on retry — show the
            // error state immediately. Retry transient failures up to 3x.
            retry: (failureCount, error) => {
              const status = (error as { status?: number } | null)?.status;
              if (typeof status === "number" && status >= 400 && status < 500) {
                return false;
              }
              return failureCount < 3;
            },
          },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      {children}
      <Toaster />
    </QueryClientProvider>
  );
}
