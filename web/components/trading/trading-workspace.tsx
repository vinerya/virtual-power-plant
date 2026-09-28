"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Eye } from "lucide-react";
import { getPortfolio, listMarkets } from "@/lib/api/trading";
import { useRole } from "@/lib/api/session";
import { MarketsPanel } from "./markets-panel";
import { OrderTicket } from "./order-ticket";
import { OrdersPanel } from "./orders-panel";
import { PortfolioPanel } from "./portfolio-panel";
import { VenueBadge } from "./trading-nav";
import { TRADING_KEYS, useMarketStream } from "./use-market-stream";

export function TradingWorkspace() {
  const { canOperate, isLoading: roleLoading, role } = useRole();
  const { history, lastTick } = useMarketStream();
  const markets = useQuery({
    queryKey: TRADING_KEYS.markets,
    queryFn: listMarkets,
    // The WebSocket keeps this fresh; poll slowly as a safety net.
    refetchInterval: 60_000,
  });
  const portfolio = useQuery({
    queryKey: TRADING_KEYS.portfolio,
    queryFn: getPortfolio,
    refetchInterval: 30_000,
  });
  const [selected, setSelected] = useState<string | undefined>();

  useEffect(() => {
    if (!selected && markets.data?.length) setSelected(markets.data[0].market);
  }, [markets.data, selected]);

  const venue = portfolio.data?.venue ?? markets.data?.[0]?.venue;

  return (
    <div className="space-y-4" data-testid="trading-workspace">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <VenueBadge venue={venue} />
        <span className="text-muted-foreground">
          Orders never leave this server: fills, prices and P&amp;L come from the built-in
          market simulator.
        </span>
      </div>

      {!roleLoading && !canOperate && (
        <p
          className="flex items-start gap-2 rounded-md border border-dashed p-3 text-xs text-muted-foreground"
          data-testid="read-only-note"
        >
          <Eye className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
          Read-only view{role ? ` (${role})` : ""}: placing and cancelling orders requires the
          operator or admin role.
        </p>
      )}

      <div
        className={
          canOperate ? "grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]" : "grid gap-4"
        }
      >
        <div className="min-w-0 space-y-4">
          <MarketsPanel
            markets={markets.data}
            isLoading={markets.isLoading}
            error={markets.error}
            onRetry={() => void markets.refetch()}
            selected={selected}
            onSelect={setSelected}
            history={history}
            lastTick={lastTick}
          />
          <PortfolioPanel
            portfolio={portfolio.data}
            isLoading={portfolio.isLoading}
            error={portfolio.error}
            onRetry={() => void portfolio.refetch()}
          />
          <OrdersPanel canTrade={canOperate} />
        </div>
        {canOperate && (
          <div>
            <div className="xl:sticky xl:top-4">
              {markets.data && markets.data.length > 0 ? (
                <OrderTicket
                  markets={markets.data}
                  market={selected}
                  onMarketChange={setSelected}
                />
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
