"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  OPEN_ORDER_STATUSES,
  cancelOrder,
  listOrders,
  listTrades,
  tradingError,
  type Order,
  type Trade,
} from "@/lib/api/trading";
import { apiErrorMessage } from "@/lib/api/errors";
import { cn, formatDateTime, formatNumber, parseApiDate } from "@/lib/utils";
import { TRADING_KEYS } from "./use-market-stream";

function SideText({ side }: { side: string }) {
  return (
    <span
      className={cn(
        "font-semibold uppercase",
        side === "buy" ? "text-emerald-600" : side === "sell" ? "text-red-600" : "",
      )}
    >
      {side}
    </span>
  );
}

export function OrderStatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase();
  const variant =
    s === "filled"
      ? "success"
      : s === "rejected"
        ? "destructive"
        : s === "pending" || s === "partial"
          ? "default"
          : "outline";
  return <Badge variant={variant}>{status}</Badge>;
}

/** Open orders, full order history and the trades blotter. */
export function OrdersPanel({ canTrade }: { canTrade: boolean }) {
  const [tab, setTab] = useState("open");
  const orders = useQuery({
    queryKey: TRADING_KEYS.orders,
    queryFn: () => listOrders({ limit: 200 }),
    refetchInterval: 30_000,
  });
  const trades = useQuery({
    queryKey: TRADING_KEYS.trades,
    queryFn: () => listTrades(100),
    refetchInterval: 30_000,
  });

  const open = (orders.data ?? []).filter((o) => OPEN_ORDER_STATUSES.has(o.status));

  return (
    <Card data-testid="orders-panel">
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-foreground">Orders &amp; trades</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList aria-label="Orders and trades">
            <TabsTrigger value="open" data-testid="tab-open-orders">
              Open orders ({open.length})
            </TabsTrigger>
            <TabsTrigger value="all">All orders</TabsTrigger>
            <TabsTrigger value="trades" data-testid="tab-trades">
              Trades
            </TabsTrigger>
          </TabsList>
          <TabsContent value="open">
            <QueryBody q={orders} what="orders">
              <OrdersTable
                rows={open}
                canTrade={canTrade}
                empty="No resting orders."
                testId="open-orders"
              />
            </QueryBody>
          </TabsContent>
          <TabsContent value="all">
            <QueryBody q={orders} what="orders">
              <OrdersTable
                rows={orders.data ?? []}
                canTrade={canTrade}
                empty="No orders yet."
                testId="all-orders"
              />
            </QueryBody>
          </TabsContent>
          <TabsContent value="trades">
            <QueryBody q={trades} what="trades">
              <TradesTable rows={trades.data ?? []} />
            </QueryBody>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function QueryBody({
  q,
  what,
  children,
}: {
  q: { isLoading: boolean; error: unknown; refetch: () => unknown };
  what: string;
  children: React.ReactNode;
}) {
  if (q.isLoading) return <Skeleton className="mt-3 h-24 w-full" />;
  if (q.error)
    return (
      <ErrorState
        className="mt-3"
        title={`Failed to load ${what}.`}
        error={q.error}
        onRetry={() => void q.refetch()}
      />
    );
  return <>{children}</>;
}

function OrdersTable({
  rows,
  canTrade,
  empty,
  testId,
}: {
  rows: Order[];
  canTrade: boolean;
  empty: string;
  testId: string;
}) {
  const qc = useQueryClient();
  const cancel = useMutation({
    mutationFn: cancelOrder,
    onSuccess: (o) => {
      toast.success(`Order ${o.id.slice(0, 8)} ${o.status}`);
    },
    onError: (err) => {
      const t = tradingError(err);
      toast.error("Cancel failed", {
        description: t?.message ?? apiErrorMessage(err, "The order could not be cancelled."),
      });
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: TRADING_KEYS.orders });
      void qc.invalidateQueries({ queryKey: TRADING_KEYS.portfolio });
    },
  });

  if (!rows.length) return <p className="py-6 text-sm text-muted-foreground">{empty}</p>;
  return (
    <div className="mt-2 max-h-80 overflow-auto">
      <table className="w-full text-sm" data-testid={testId}>
        <thead className="sticky top-0 border-b bg-card text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Created</th>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Market</th>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Side</th>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Type</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Qty</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Filled</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Price</th>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Status</th>
            {canTrade && (
              <th scope="col" className="py-2 text-right font-medium">
                <span className="sr-only">Actions</span>
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((o) => {
            const reasons = o.metadata?.reject_reasons;
            const cancellable = OPEN_ORDER_STATUSES.has(o.status);
            return (
              <tr key={o.id} className="border-b last:border-b-0" data-testid={`order-${o.id}`}>
                <td className="py-2 pr-3 text-xs tabular-nums text-muted-foreground">
                  {formatDateTime(parseApiDate(o.created_at))}
                </td>
                <td className="py-2 pr-3">{o.market}</td>
                <td className="py-2 pr-3">
                  <SideText side={o.side} />
                </td>
                <td className="py-2 pr-3 text-muted-foreground">
                  {o.order_type}
                  <span className="ml-1 text-[10px]">{o.time_in_force}</span>
                </td>
                <td className="py-2 pr-3 text-right tabular-nums">{formatNumber(o.quantity, 1)}</td>
                <td className="py-2 pr-3 text-right tabular-nums">
                  {formatNumber(o.filled_quantity, 1)}
                </td>
                <td className="py-2 pr-3 text-right tabular-nums">
                  {o.price > 0 ? formatNumber(o.price) : "mkt"}
                  {o.average_price > 0 && (
                    <span className="block text-[10px] text-muted-foreground">
                      avg {formatNumber(o.average_price)}
                    </span>
                  )}
                </td>
                <td className="py-2 pr-3">
                  <OrderStatusBadge status={o.status} />
                  {Array.isArray(reasons) && reasons.length > 0 && (
                    <span className="block max-w-[16rem] text-[10px] text-destructive">
                      {reasons.map(String).join("; ")}
                    </span>
                  )}
                </td>
                {canTrade && (
                  <td className="py-2 text-right">
                    {cancellable && (
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={cancel.isPending && cancel.variables === o.id}
                        onClick={() => cancel.mutate(o.id)}
                        aria-label={`Cancel ${o.side} ${o.quantity} ${o.market} order`}
                      >
                        Cancel
                      </Button>
                    )}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradesTable({ rows }: { rows: Trade[] }) {
  if (!rows.length) return <p className="py-6 text-sm text-muted-foreground">No fills yet.</p>;
  return (
    <div className="mt-2 max-h-80 overflow-auto">
      <table className="w-full text-sm" data-testid="trades-blotter">
        <thead className="sticky top-0 border-b bg-card text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Time</th>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Market</th>
            <th scope="col" className="py-2 pr-3 text-left font-medium">Side</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Qty</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Price</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Fees</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Realized P&amp;L</th>
            <th scope="col" className="py-2 text-left font-medium">Strategy</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <tr key={t.id} className="border-b last:border-b-0">
              <td className="py-2 pr-3 text-xs tabular-nums text-muted-foreground">
                {formatDateTime(parseApiDate(t.timestamp))}
              </td>
              <td className="py-2 pr-3">{t.market}</td>
              <td className="py-2 pr-3">
                <SideText side={t.side} />
              </td>
              <td className="py-2 pr-3 text-right tabular-nums">{formatNumber(t.quantity, 1)}</td>
              <td className="py-2 pr-3 text-right tabular-nums">{formatNumber(t.price)}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-muted-foreground">
                {formatNumber(t.fees)}
              </td>
              <td
                className={cn(
                  "py-2 pr-3 text-right tabular-nums",
                  t.realized_pnl > 0 && "text-emerald-600",
                  t.realized_pnl < 0 && "text-destructive",
                )}
              >
                {formatNumber(t.realized_pnl)}
              </td>
              <td className="py-2 text-xs text-muted-foreground">{t.strategy ?? "manual"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
