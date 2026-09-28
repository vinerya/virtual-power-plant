"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { SavingsCard } from "@/components/customer/savings-card";
import { EnergyFlow } from "@/components/customer/energy-flow";
import { BillUnavailable, isBillUnavailable } from "@/components/customer/bill-unavailable";
import { getMe, getMyBill, getMyDevices } from "@/lib/api/customer";
import { ErrorState } from "@/components/ui/error-state";

export default function PortalHomePage() {
  const me = useQuery({ queryKey: ["customer", "me"], queryFn: getMe });
  const bill = useQuery({
    queryKey: ["customer", "bill"],
    queryFn: () => getMyBill(),
  });
  const devices = useQuery({
    queryKey: ["customer", "devices"],
    queryFn: getMyDevices,
    refetchInterval: 15_000,
  });

  if (me.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-1/3" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  if (!me.data) {
    return (
      <ErrorState
        title="Failed to load your portal."
        error={me.error}
        onRetry={() => void me.refetch()}
      />
    );
  }

  const total = bill.data?.bill.total;
  const baseline = bill.data ? (bill.data.baseline?.total ?? bill.data.bill.total) : undefined;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          Welcome back, {firstName(me.data.name)}.
        </h2>
        <p className="text-sm text-muted-foreground">
          Here is what your home is doing this month.
        </p>
      </div>

      {bill.isLoading ? (
        <Skeleton className="h-32 w-full" />
      ) : bill.data && total != null && baseline != null ? (
        <SavingsCard
          thisMonth={total}
          baseline={baseline}
          thisMonthKwh={bill.data.this_month_kwh}
          lastMonthKwh={bill.data.last_month_kwh}
        />
      ) : isBillUnavailable(bill.error) ? (
        <BillUnavailable error={bill.error} />
      ) : (
        <ErrorState
          title="Failed to load your bill."
          error={bill.error}
          onRetry={() => void bill.refetch()}
        />
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {devices.isLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : devices.data ? (
          <EnergyFlow devices={devices.data} />
        ) : (
          <ErrorState
            title="Failed to load live device data."
            error={devices.error}
            onRetry={() => void devices.refetch()}
          />
        )}
        {bill.data && total != null && baseline != null && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>This month vs. last</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <Row
                label="Bill"
                left={`$${total.toFixed(2)}`}
                right={`$${baseline.toFixed(2)}`}
              />
              <Row
                label="Energy used"
                left={`${bill.data.this_month_kwh ?? "—"} kWh`}
                right={`${bill.data.last_month_kwh ?? "—"} kWh`}
              />
              <p className="pt-2 text-xs text-muted-foreground">
                Compared to your typical month at the same usage.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function firstName(name: string) {
  return name.split(/\s+/)[0] ?? name;
}

function Row({
  label,
  left,
  right,
}: {
  label: string;
  left: string;
  right: string;
}) {
  return (
    <div className="grid grid-cols-3 items-baseline gap-2 border-b pb-2 last:border-b-0 last:pb-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium tabular-nums">{left}</span>
      <span className="text-right text-muted-foreground tabular-nums">
        {right}
      </span>
    </div>
  );
}
