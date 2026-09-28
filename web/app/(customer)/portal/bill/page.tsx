"use client";

import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import { BillBreakdown } from "@/components/tariffs/bill-breakdown";
import { getMyBill } from "@/lib/api/customer";
import { ErrorState } from "@/components/ui/error-state";
import { BillUnavailable, isBillUnavailable } from "@/components/customer/bill-unavailable";

export default function CustomerBillPage() {
  const q = useQuery({
    queryKey: ["customer", "bill"],
    queryFn: () => getMyBill(),
  });

  return (
    <div className="space-y-6" data-testid="customer-bill-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Your bill</h2>
        <p className="text-sm text-muted-foreground">
          Read-only breakdown of this month&apos;s charges.
        </p>
      </div>

      {q.isLoading ? (
        <Skeleton className="h-72 w-full" />
      ) : isBillUnavailable(q.error) ? (
        <BillUnavailable error={q.error} />
      ) : q.isError || !q.data ? (
        <ErrorState
          title="Failed to load your bill."
          error={q.error}
          onRetry={() => q.refetch()}
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          <BillBreakdown bill={q.data.bill} title="This month" />
          {q.data.baseline && (
            <BillBreakdown
              bill={q.data.baseline}
              title="Baseline (typical)"
              comparison={q.data.bill}
            />
          )}
        </div>
      )}
    </div>
  );
}
