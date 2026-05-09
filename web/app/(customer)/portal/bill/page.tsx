"use client";

import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import { BillBreakdown } from "@/components/tariffs/bill-breakdown";
import { getMyBill } from "@/lib/api/customer";

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
      ) : q.isError || !q.data ? (
        <p className="text-sm text-destructive">Failed to load bill.</p>
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
