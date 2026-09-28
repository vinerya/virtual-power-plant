"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { asBusEvent, getWsClient } from "@/lib/ws/client";
import { marketSchema, mergeMarket, type Market } from "@/lib/api/trading";

export const TRADING_KEYS = {
  markets: ["trading", "markets"] as const,
  orders: ["trading", "orders"] as const,
  trades: ["trading", "trades"] as const,
  portfolio: ["trading", "portfolio"] as const,
};

const ORDER_EVENTS = new Set([
  "order_submitted",
  "order_filled",
  "order_cancelled",
  "order_rejected",
  "trade_executed",
]);

const HISTORY = 120;

export interface PricePoint {
  t: number;
  price: number;
}

/**
 * Subscribes to the `market_data` WebSocket channel while mounted.
 *
 * - `market_data` events carry a market snapshot (same shape as
 *   GET /trading/markets rows): merged into the markets query cache and
 *   appended to an in-memory price history (since the page was opened).
 * - order/trade events invalidate orders, trades and the portfolio.
 */
export function useMarketStream() {
  const qc = useQueryClient();
  const [history, setHistory] = useState<Record<string, PricePoint[]>>({});
  const [lastTick, setLastTick] = useState<number | null>(null);
  const pending = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const client = getWsClient();
    client.subscribe("market_data");
    const off = client.on((msg) => {
      if (msg.channel !== "market_data") return;
      const ev = asBusEvent(msg.data);
      const type = ev?.event_type ?? "market_data";
      const payload = ev ? ev.data : msg.data;
      if (type === "market_data") {
        const parsed = marketSchema.safeParse(payload);
        if (!parsed.success) return;
        const snap: Market = parsed.data;
        qc.setQueryData<Market[]>(TRADING_KEYS.markets, (cur) => mergeMarket(cur, snap));
        const now = Date.now();
        setLastTick(now);
        if (snap.last_price != null) {
          const price = snap.last_price;
          setHistory((h) => {
            const prev = h[snap.market] ?? [];
            return { ...h, [snap.market]: [...prev, { t: now, price }].slice(-HISTORY) };
          });
        }
        // Marks move with prices: refresh unrealized P&L, coalesced.
        if (!pending.current) {
          pending.current = setTimeout(() => {
            pending.current = null;
            void qc.invalidateQueries({ queryKey: TRADING_KEYS.portfolio });
          }, 2_000);
        }
        return;
      }
      if (ORDER_EVENTS.has(type)) {
        void qc.invalidateQueries({ queryKey: TRADING_KEYS.orders });
        void qc.invalidateQueries({ queryKey: TRADING_KEYS.trades });
        void qc.invalidateQueries({ queryKey: TRADING_KEYS.portfolio });
      }
    });
    return () => {
      off();
      client.unsubscribe("market_data");
      if (pending.current) clearTimeout(pending.current);
      pending.current = null;
    };
  }, [qc]);

  return { history, lastTick };
}
