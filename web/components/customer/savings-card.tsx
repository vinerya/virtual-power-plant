"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingDown, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

export function SavingsCard({
  thisMonth,
  baseline,
  thisMonthKwh,
  lastMonthKwh,
}: {
  thisMonth: number;
  baseline: number;
  thisMonthKwh?: number;
  lastMonthKwh?: number;
}) {
  const saved = baseline - thisMonth;
  const savedPct = baseline > 0 ? (saved / baseline) * 100 : 0;
  const positive = saved >= 0;

  const usageDelta =
    thisMonthKwh != null && lastMonthKwh != null
      ? thisMonthKwh - lastMonthKwh
      : null;

  return (
    <Card data-testid="savings-card">
      <CardHeader className="pb-2">
        <CardTitle>Savings this month</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            vs. baseline
          </p>
          <p
            className={cn(
              "mt-1 text-3xl font-semibold tabular-nums",
              positive ? "text-emerald-600" : "text-destructive",
            )}
            data-testid="savings-amount"
          >
            {positive ? "−" : "+"}${Math.abs(saved).toFixed(2)}
          </p>
          <p className="text-xs text-muted-foreground">
            {savedPct.toFixed(1)}% off your typical bill
          </p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            This month
          </p>
          <p className="mt-1 text-2xl font-semibold tabular-nums">
            ${thisMonth.toFixed(2)}
          </p>
          <p className="text-xs text-muted-foreground">vs. ${baseline.toFixed(2)} typical</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Energy used
          </p>
          <p className="mt-1 text-2xl font-semibold tabular-nums">
            {thisMonthKwh != null ? `${thisMonthKwh} kWh` : "—"}
          </p>
          {usageDelta != null && (
            <p className="flex items-center gap-1 text-xs text-muted-foreground">
              {usageDelta < 0 ? (
                <TrendingDown className="h-3 w-3 text-emerald-500" />
              ) : (
                <TrendingUp className="h-3 w-3 text-amber-500" />
              )}
              {Math.abs(usageDelta)} kWh vs. last month
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
