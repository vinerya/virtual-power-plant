"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/ui/error-state";
import type { Market } from "@/lib/api/trading";
import { cn, formatNumber, formatRelativeTime, parseApiDate } from "@/lib/utils";
import { PriceTrace } from "./lazy-charts";
import type { PricePoint } from "./use-market-stream";

export function MarketsPanel({
  markets,
  isLoading,
  error,
  onRetry,
  selected,
  onSelect,
  history,
  lastTick,
}: {
  markets: Market[] | undefined;
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
  selected: string | undefined;
  onSelect: (m: string) => void;
  history: Record<string, PricePoint[]>;
  lastTick: number | null;
}) {
  const sel = markets?.find((m) => m.market === selected);
  const trace = selected ? (history[selected] ?? []) : [];

  return (
    <Card data-testid="markets-panel">
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-base text-foreground">Markets</CardTitle>
        <span className="text-xs text-muted-foreground" aria-live="polite">
          {lastTick ? `live tick ${formatRelativeTime(new Date(lastTick))}` : "waiting for live ticks"}
        </span>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <Skeleton className="h-24 w-full" />
        ) : error ? (
          <ErrorState title="Failed to load markets." error={error} onRetry={onRetry} />
        ) : !markets?.length ? (
          <p className="text-sm text-muted-foreground">No markets are configured.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">
                Latest quotes per market, prices in {markets[0].price_unit}
              </caption>
              <thead className="border-b text-xs uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th scope="col" className="py-2 pr-3 text-left font-medium">Market</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Last</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Bid</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Ask</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Spread</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Volume</th>
                  <th scope="col" className="py-2 text-left font-medium">Updated</th>
                </tr>
              </thead>
              <tbody data-testid="market-rows">
                {markets.map((m) => {
                  const spread = m.bid != null && m.ask != null ? m.ask - m.bid : null;
                  const active = m.market === selected;
                  return (
                    <tr
                      key={m.market}
                      data-testid={`market-row-${m.market}`}
                      className={cn("border-b last:border-b-0", active && "bg-primary/5")}
                    >
                      <th scope="row" className="py-2 pr-3 text-left font-medium">
                        <button
                          type="button"
                          onClick={() => onSelect(m.market)}
                          aria-pressed={active}
                          className="rounded text-left underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                          {m.market}
                        </button>
                        <Badge
                          variant={m.status === "open" ? "success" : "outline"}
                          className="ml-2 align-middle"
                        >
                          {m.status}
                        </Badge>
                      </th>
                      <td
                        className="py-2 pr-3 text-right font-semibold tabular-nums"
                        data-testid={`last-${m.market}`}
                      >
                        {formatNumber(m.last_price)}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums">{formatNumber(m.bid)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">{formatNumber(m.ask)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums text-muted-foreground">
                        {formatNumber(spread)}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums text-muted-foreground">
                        {formatNumber(m.volume, 1)}
                      </td>
                      <td className="py-2 text-xs text-muted-foreground">
                        {formatRelativeTime(parseApiDate(m.timestamp))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-muted-foreground">
              Prices {markets[0].price_unit}, quantities {markets[0].quantity_unit}. Source:{" "}
              {Array.from(new Set(markets.map((m) => m.source))).join(", ")}.
            </p>
          </div>
        )}

        {sel && (
          <section aria-label={`${sel.market} live price`}>
            <h3 className="mb-1 text-xs font-medium text-muted-foreground">
              {sel.market} last price since this page opened
            </h3>
            {trace.length >= 2 ? (
              <PriceTrace
                data={trace}
                unit={sel.price_unit}
                label={`${sel.market} last price, ${trace.length} live ticks`}
              />
            ) : (
              <p className="rounded-md border border-dashed p-4 text-xs text-muted-foreground">
                The trace fills in as live ticks arrive on the market data stream.
              </p>
            )}
          </section>
        )}
      </CardContent>
    </Card>
  );
}
