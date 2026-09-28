// Trading API (src/vpp/api/routes/trading.py, src/vpp/schemas/trading.py).
//
// The backend venue is SIMULATED: orders never leave the API process and
// every market carries `venue: "simulated"`. Prices are $/MWh, quantities
// MWh. Datetimes are ISO strings, sometimes without a timezone suffix
// (naive UTC), so they are typed as plain strings.
import { z } from "zod";
import { api } from "./client";
import { apiDetail, apiStatus, parseResponse } from "./errors";
import { withMockFallback } from "./mocks";

const num = z.number();
const optNum = z.number().nullable().optional();

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

export const marketSchema = z.object({
  market: z.string(),
  market_type: z.string(),
  status: z.string(),
  venue: z.string().default("simulated"),
  source: z.string().default("simulated"),
  currency: z.string().default("USD"),
  price_unit: z.string().default("$/MWh"),
  quantity_unit: z.string().default("MWh"),
  tick_size: num,
  lot_size: num,
  fee_per_unit: num,
  last_price: optNum,
  bid: optNum,
  ask: optNum,
  volume: num.default(0),
  timestamp: z.string().nullable().optional(),
  depth: z
    .record(z.array(z.tuple([num, num])))
    .nullable()
    .optional(),
});
export type Market = z.infer<typeof marketSchema>;

export const orderSchema = z.object({
  id: z.string(),
  order_type: z.string(),
  market: z.string(),
  side: z.string(),
  quantity: num,
  price: num,
  status: z.string(),
  filled_quantity: num.default(0),
  remaining_quantity: num.default(0),
  average_price: num.default(0),
  time_in_force: z.string().default("GTC"),
  created_at: z.string(),
  updated_at: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).default({}),
});
export type Order = z.infer<typeof orderSchema>;

export const tradeSchema = z.object({
  id: z.string(),
  order_id: z.string(),
  market: z.string(),
  side: z.string(),
  quantity: num,
  price: num,
  fees: num.default(0),
  timestamp: z.string(),
  strategy: z.string().nullable().optional(),
  realized_pnl: num.default(0),
});
export type Trade = z.infer<typeof tradeSchema>;

export const orderSubmitSchema = orderSchema.extend({
  fills: z.array(tradeSchema).default([]),
});
export type OrderSubmitResult = z.infer<typeof orderSubmitSchema>;

const positionSchema = z.object({
  market: z.string(),
  quantity: num,
  average_price: num,
  mark_price: optNum,
  unrealized_pnl: num.default(0),
  realized_pnl: num.default(0),
  notional_value: num.default(0),
});
export type Position = z.infer<typeof positionSchema>;

const riskLimitsSchema = z.object({
  max_position: num,
  max_daily_loss: num,
  max_drawdown: num,
  var_limit: num,
  concentration_limit: num,
});

const riskSchema = z.object({
  var_95_1d: optNum,
  var_method: z.string().default(""),
  daily_pnl: num.default(0),
  concentrations: z.record(num).default({}),
  breach: z.boolean().default(false),
  breaches: z.array(z.string()).default([]),
  volatility: z
    .record(z.object({ value: num, source: z.string(), samples: num.default(0) }))
    .default({}),
  limits: riskLimitsSchema.nullable().optional(),
});
export type RiskSummary = z.infer<typeof riskSchema>;

export const portfolioSchema = z.object({
  cash: num,
  equity: num,
  total_pnl: num,
  max_drawdown: num,
  positions: z.array(positionSchema).default([]),
  total_trades: num.default(0),
  last_updated: z.string(),
  initial_cash: num.default(0),
  realized_pnl: num.default(0),
  unrealized_pnl: num.default(0),
  fees_paid: num.default(0),
  current_drawdown: num.default(0),
  gross_exposure: num.default(0),
  net_exposure: num.default(0),
  open_orders: num.default(0),
  risk: riskSchema.nullable().optional(),
  venue: z.string().default("simulated"),
});
export type Portfolio = z.infer<typeof portfolioSchema>;

export const strategySchema = z.object({
  name: z.string(),
  description: z.string(),
  parameters: z.record(z.unknown()),
  min_markets: num.default(1),
});
export type Strategy = z.infer<typeof strategySchema>;

export const strategyBacktestSchema = z.object({
  strategy: z.string(),
  data_source: z.string(),
  markets: z.array(z.string()),
  periods: num,
  interval_minutes: num,
  initial_cash: num,
  final_equity: num,
  total_pnl: num,
  total_return: num,
  realized_pnl: num,
  unrealized_pnl: num,
  fees: num,
  sharpe_ratio: num,
  max_drawdown: num,
  num_trades: num,
  win_rate: optNum,
  final_positions: z.record(num).default({}),
  equity_curve: z.array(z.object({ timestamp: z.string(), equity: num })).default([]),
  trades: z
    .array(
      z.object({
        timestamp: z.string(),
        market: z.string(),
        side: z.string(),
        quantity: num,
        price: num,
        fees: num,
        realized_pnl: num,
      }),
    )
    .default([]),
  assumptions: z.array(z.string()).default([]),
});
export type StrategyBacktest = z.infer<typeof strategyBacktestSchema>;

/** 422/404/409 body from the trading service (`TradingError.to_detail`). */
export const tradingErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  reasons: z.array(z.string()).default([]),
  order_id: z.string().optional(),
});
export type TradingErrorDetail = z.infer<typeof tradingErrorSchema>;

/** Structured trading rejection, or null if `err` is something else. */
export function tradingError(err: unknown): (TradingErrorDetail & { status?: number }) | null {
  const parsed = tradingErrorSchema.safeParse(apiDetail(err));
  return parsed.success ? { ...parsed.data, status: apiStatus(err) } : null;
}

// ---------------------------------------------------------------------------
// Order ticket
// ---------------------------------------------------------------------------

export const ORDER_TYPES = [
  "market",
  "limit",
  "stop",
  "stop_limit",
  "iceberg",
  "ioc",
  "fok",
] as const;
export type OrderType = (typeof ORDER_TYPES)[number];
export const TIME_IN_FORCE = ["GTC", "DAY", "IOC", "FOK"] as const;

export interface OrderCreate {
  order_type: OrderType;
  market: string;
  side: "buy" | "sell";
  quantity: number;
  price?: number;
  stop_price?: number;
  limit_price?: number;
  visible_quantity?: number;
  time_in_force?: (typeof TIME_IN_FORCE)[number];
}

/** Resting orders that can still be cancelled. */
export const OPEN_ORDER_STATUSES = new Set(["pending", "partial", "open", "partially_filled"]);

// ---------------------------------------------------------------------------
// Calls
// ---------------------------------------------------------------------------

const P = "/api/v1/trading";

export async function listMarkets(): Promise<Market[]> {
  return withMockFallback(
    async () => parseResponse(z.array(marketSchema), await api.get(`${P}/markets`), "markets"),
    () => MOCK_MARKETS,
  );
}

export async function listOrders(params: { status?: string; limit?: number } = {}): Promise<Order[]> {
  const q = new URLSearchParams();
  q.set("limit", String(params.limit ?? 100));
  if (params.status) q.set("order_status", params.status);
  return withMockFallback(
    async () => parseResponse(z.array(orderSchema), await api.get(`${P}/orders?${q}`), "orders"),
    () => [],
  );
}

export async function listTrades(limit = 100): Promise<Trade[]> {
  return withMockFallback(
    async () =>
      parseResponse(z.array(tradeSchema), await api.get(`${P}/trades?limit=${limit}`), "trades"),
    () => [],
  );
}

export async function getPortfolio(): Promise<Portfolio> {
  return withMockFallback(
    async () => parseResponse(portfolioSchema, await api.get(`${P}/portfolio`), "portfolio"),
    () => MOCK_PORTFOLIO,
  );
}

export async function submitOrder(body: OrderCreate): Promise<OrderSubmitResult> {
  // No mock fallback: in demo mode an order must not appear to succeed.
  return parseResponse(orderSubmitSchema, await api.post(`${P}/orders`, body), "order");
}

export async function cancelOrder(id: string): Promise<Order> {
  return parseResponse(
    orderSchema,
    await api.delete(`${P}/orders/${encodeURIComponent(id)}`),
    "order",
  );
}

export async function listStrategies(): Promise<Strategy[]> {
  return withMockFallback(
    async () =>
      parseResponse(z.array(strategySchema), await api.get(`${P}/strategies`), "strategies"),
    () => MOCK_STRATEGIES,
  );
}

export interface StrategyBacktestRequest {
  params?: Record<string, unknown>;
  synthetic?: { periods?: number; interval_minutes?: number; seed?: number };
  initial_cash?: number;
  fee_per_unit?: number;
  half_spread?: number;
}

export async function backtestStrategy(
  name: string,
  body: StrategyBacktestRequest,
): Promise<StrategyBacktest> {
  return parseResponse(
    strategyBacktestSchema,
    await api.post(`${P}/strategies/${encodeURIComponent(name)}/backtest`, body),
    "backtest",
  );
}

/** Merge one live market snapshot (WebSocket `market_data`) into a list. */
export function mergeMarket(list: Market[] | undefined, snap: Market): Market[] {
  const cur = list ?? [];
  const idx = cur.findIndex((m) => m.market === snap.market);
  if (idx === -1) return [...cur, snap];
  const next = cur.slice();
  next[idx] = { ...cur[idx], ...snap, depth: snap.depth ?? cur[idx].depth };
  return next;
}

// ---------------------------------------------------------------------------
// Demo data (NEXT_PUBLIC_USE_MOCKS=1 only)
// ---------------------------------------------------------------------------

const MOCK_TS = "2026-01-01T12:00:00Z";

const MOCK_MARKETS: Market[] = [
  {
    market: "day_ahead",
    market_type: "day_ahead",
    status: "open",
    venue: "simulated",
    source: "simulated",
    currency: "USD",
    price_unit: "$/MWh",
    quantity_unit: "MWh",
    tick_size: 0.01,
    lot_size: 0.1,
    fee_per_unit: 0.07,
    last_price: 55.6,
    bid: 55.49,
    ask: 55.71,
    volume: 113,
    timestamp: MOCK_TS,
  },
  {
    market: "real_time",
    market_type: "real_time",
    status: "open",
    venue: "simulated",
    source: "simulated",
    currency: "USD",
    price_unit: "$/MWh",
    quantity_unit: "MWh",
    tick_size: 0.01,
    lot_size: 0.1,
    fee_per_unit: 0.07,
    last_price: 61.44,
    bid: 61.33,
    ask: 61.55,
    volume: 121.5,
    timestamp: MOCK_TS,
  },
];

const MOCK_PORTFOLIO: Portfolio = {
  cash: 100_000,
  equity: 100_000,
  total_pnl: 0,
  max_drawdown: 0,
  positions: [],
  total_trades: 0,
  last_updated: MOCK_TS,
  initial_cash: 100_000,
  realized_pnl: 0,
  unrealized_pnl: 0,
  fees_paid: 0,
  current_drawdown: 0,
  gross_exposure: 0,
  net_exposure: 0,
  open_orders: 0,
  risk: {
    var_95_1d: 0,
    var_method: "parametric (delta-normal), 1-day horizon, 95% one-sided",
    daily_pnl: 0,
    concentrations: {},
    breach: false,
    breaches: [],
    volatility: {},
    limits: {
      max_position: 50,
      max_daily_loss: 5000,
      max_drawdown: 0.2,
      var_limit: 10000,
      concentration_limit: 0.8,
    },
  },
  venue: "simulated",
};

const MOCK_STRATEGIES: Strategy[] = [
  {
    name: "momentum",
    description:
      "Trend following: buys when the price rose more than momentum_threshold over the lookback window, sells when it fell.",
    parameters: {
      lookback_hours: 4,
      momentum_threshold: 0.05,
      base_quantity: 10,
      max_position_size: 50,
      max_daily_trades: 24,
    },
    min_markets: 1,
  },
];
