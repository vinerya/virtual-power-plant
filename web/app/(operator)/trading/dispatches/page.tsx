import { Suspense } from "react";
import { DispatchesView } from "@/components/dispatch/dispatches-view";
import { TradingNav } from "@/components/trading/trading-nav";
import { Skeleton } from "@/components/ui/skeleton";

export default function DispatchesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Dispatch history</h2>
        <p className="text-sm text-muted-foreground">
          Past optimization runs. Click any row for inputs, schedule, and
          diagnostics.
        </p>
      </div>
      <TradingNav />
      {/* useSearchParams (the ?run= deep link) needs a Suspense boundary. */}
      <Suspense fallback={<Skeleton className="h-64 w-full" />}>
        <DispatchesView />
      </Suspense>
    </div>
  );
}
