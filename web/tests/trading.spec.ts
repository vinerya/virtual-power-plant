import { test, expect } from "@playwright/test";
import { api, json, signInAs } from "./support";

const TS = "2026-09-28T12:00:00";

const market = (name: string, last: number) => ({
  market: name,
  market_type: name,
  status: "open",
  venue: "simulated",
  source: "simulated",
  currency: "USD",
  price_unit: "$/MWh",
  quantity_unit: "MWh",
  tick_size: 0.01,
  lot_size: 0.1,
  fee_per_unit: 0.07,
  last_price: last,
  bid: last - 0.1,
  ask: last + 0.1,
  volume: 100,
  timestamp: TS,
  depth: null,
});

const MARKETS = [market("day_ahead", 55.6), market("real_time", 61.4)];

const ORDERS = [
  {
    id: "ord-resting-1",
    order_type: "limit",
    market: "day_ahead",
    side: "buy",
    quantity: 5,
    price: 40,
    status: "pending",
    filled_quantity: 0,
    remaining_quantity: 5,
    average_price: 0,
    time_in_force: "GTC",
    created_at: TS,
    updated_at: TS,
    metadata: { submitted_by: "operator" },
  },
  {
    id: "ord-filled-1",
    order_type: "market",
    market: "real_time",
    side: "sell",
    quantity: 3,
    price: 0,
    status: "filled",
    filled_quantity: 3,
    remaining_quantity: 0,
    average_price: 61.2,
    time_in_force: "IOC",
    created_at: TS,
    updated_at: TS,
    metadata: {},
  },
];

const TRADES = [
  {
    id: "t1",
    order_id: "ord-filled-1",
    market: "real_time",
    side: "sell",
    quantity: 3,
    price: 61.2,
    fees: 0.21,
    timestamp: TS,
    strategy: null,
    realized_pnl: 4.5,
  },
];

const PORTFOLIO = {
  cash: 100_183.6,
  equity: 99_990,
  total_pnl: -10,
  max_drawdown: 0.001,
  positions: [
    {
      market: "real_time",
      quantity: -3,
      average_price: 61.2,
      mark_price: 61.4,
      unrealized_pnl: -0.6,
      realized_pnl: 0,
      notional_value: 184.2,
    },
  ],
  total_trades: 1,
  last_updated: TS,
  initial_cash: 100_000,
  realized_pnl: 4.5,
  unrealized_pnl: -0.6,
  fees_paid: 0.21,
  current_drawdown: 0.0001,
  gross_exposure: 184.2,
  net_exposure: -184.2,
  open_orders: 1,
  risk: {
    var_95_1d: 12000,
    var_method: "parametric (delta-normal), 1-day horizon, 95% one-sided",
    daily_pnl: -10,
    concentrations: { real_time: 1 },
    breach: true,
    breaches: ["VaR limit exceeded: 1-day 95% VaR 12000.00 > 10000.00"],
    volatility: { real_time: { value: 0.35, source: "model_prior", samples: 3 } },
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

const STRATEGIES = [
  {
    name: "momentum",
    description: "Trend following.",
    parameters: { lookback_hours: 4, momentum_threshold: 0.05, base_quantity: 10 },
    min_markets: 1,
  },
];

const BACKTEST = {
  strategy: "momentum",
  data_source: "synthetic",
  markets: ["day_ahead", "real_time"],
  periods: 4,
  interval_minutes: 60,
  initial_cash: 100000,
  final_equity: 100250,
  total_pnl: 250,
  total_return: 0.0025,
  realized_pnl: 200,
  unrealized_pnl: 60,
  fees: 10,
  sharpe_ratio: 1.23,
  max_drawdown: 0.004,
  num_trades: 6,
  win_rate: 0.5,
  final_positions: { day_ahead: 10 },
  equity_curve: [
    { timestamp: "2025-01-01T00:00:00", equity: 100000 },
    { timestamp: "2025-01-01T01:00:00", equity: 99900 },
    { timestamp: "2025-01-01T02:00:00", equity: 100100 },
    { timestamp: "2025-01-01T03:00:00", equity: 100250 },
  ],
  trades: [],
  assumptions: ["Market-order signals fill in full at last price +/- half_spread."],
};

async function stubTrading(context: import("@playwright/test").BrowserContext) {
  await context.route(api("/api/v1/trading/markets"), (r) => json(r, MARKETS));
  await context.route(api("/api/v1/trading/portfolio"), (r) => json(r, PORTFOLIO));
  await context.route(api("/api/v1/trading/trades"), (r) => json(r, TRADES));
  await context.route(api("/api/v1/trading/strategies"), (r) => json(r, STRATEGIES));
  await context.route(api("/api/v1/trading/orders"), (r) => {
    if (r.request().method() === "GET") return json(r, ORDERS);
    return r.fallback();
  });
}

test.describe("trading workspace (operator)", () => {
  test.beforeEach(async ({ context }) => {
    await signInAs(context, "operator");
    await stubTrading(context);
  });

  test("shows simulated venue, markets, portfolio risk and blotter", async ({ page }) => {
    await page.goto("/trading");
    await expect(page.getByTestId("venue-badge")).toHaveText(/simulated venue/i);
    await expect(page.getByTestId("last-day_ahead")).toHaveText("55.60");
    await expect(page.getByTestId("last-real_time")).toHaveText("61.40");

    // Portfolio: breach list + VaR over its limit.
    await expect(page.getByTestId("limit-breaches")).toContainText("VaR limit exceeded");
    await expect(page.getByTestId("stat-var")).toContainText("$12,000.00");
    await expect(page.getByTestId("positions-table")).toContainText("real_time");

    // Only resting orders are in the open tab.
    await expect(page.getByTestId("open-orders").locator("tbody tr")).toHaveCount(1);
    await page.getByTestId("tab-trades").click();
    await expect(page.getByTestId("trades-blotter")).toContainText("61.20");

    // Keyboard: arrow keys move between the order/trade tabs.
    await page.getByTestId("tab-trades").focus();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByRole("tab", { name: "All orders" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  test("submits an order and shows risk-rejection reasons", async ({ page }) => {
    const bodies: unknown[] = [];
    await page.route(api("/api/v1/trading/orders"), (r) => {
      if (r.request().method() !== "POST") return r.fallback();
      const body = r.request().postDataJSON();
      bodies.push(body);
      if (body.quantity > 50) {
        return json(
          r,
          {
            detail: {
              code: "risk_limit_breached",
              message: "Order rejected by pre-trade risk checks",
              reasons: [
                "Position limit exceeded in day_ahead: post-trade 505 > 50",
                "VaR limit exceeded: post-trade 1-day 95% VaR 13444.41 > 10000.00",
              ],
              order_id: "rejected-123456",
            },
          },
          422,
        );
      }
      return json(
        r,
        {
          ...ORDERS[0],
          id: "new-1",
          order_type: "market",
          status: "filled",
          quantity: body.quantity,
          filled_quantity: body.quantity,
          average_price: 55.7,
          fills: [
            {
              id: "f1",
              order_id: "new-1",
              market: "day_ahead",
              side: "buy",
              quantity: body.quantity,
              price: 55.7,
              fees: 0.14,
              timestamp: TS,
              realized_pnl: 0,
            },
          ],
        },
        201,
      );
    });

    await page.goto("/trading");
    const ticket = page.getByTestId("order-ticket");
    await expect(ticket).toBeVisible();

    // Labels are wired to their controls.
    await ticket.getByLabel("Order type").selectOption("market");
    await ticket.getByLabel(/Quantity/).fill("2");
    await page.getByTestId("ticket-submit").click();
    await expect(page.getByText("Order filled")).toBeVisible();
    expect(bodies[0]).toMatchObject({
      order_type: "market",
      market: "day_ahead",
      side: "buy",
      quantity: 2,
    });
    expect(bodies[0]).not.toHaveProperty("price");

    await ticket.getByLabel(/Quantity/).fill("500");
    await page.getByTestId("ticket-submit").click();
    const rejection = page.getByTestId("order-rejection");
    await expect(rejection).toContainText("Rejected by pre-trade risk checks");
    await expect(rejection).toContainText("Position limit exceeded in day_ahead");
    await expect(rejection).toContainText("VaR limit exceeded");
    await expect(rejection).toContainText("rejected");
  });

  test("validates required prices client-side", async ({ page }) => {
    let posted = 0;
    await page.route(api("/api/v1/trading/orders"), (r) => {
      if (r.request().method() === "POST") posted += 1;
      return r.fallback();
    });
    await page.goto("/trading");
    const ticket = page.getByTestId("order-ticket");
    await ticket.getByLabel("Order type").selectOption("stop_limit");
    await ticket.getByLabel("Stop trigger").fill("");
    await page.getByTestId("ticket-submit").click();
    await expect(ticket.getByText("Enter a positive stop trigger price")).toBeVisible();
    await expect(ticket.getByLabel("Stop trigger")).toHaveAttribute("aria-invalid", "true");
    expect(posted).toBe(0);
  });

  test("cancels a resting order", async ({ page }) => {
    let cancelled = "";
    await page.route(/\/api\/proxy\/api\/v1\/trading\/orders\/[^/?]+$/, (r) => {
      if (r.request().method() !== "DELETE") return r.fallback();
      cancelled = r.request().url().split("/").pop() ?? "";
      return json(r, { ...ORDERS[0], status: "cancelled" });
    });
    await page.goto("/trading");
    await page.getByRole("button", { name: /Cancel buy 5 day_ahead order/ }).click();
    await expect(page.getByText(/ord-rest cancelled/)).toBeVisible();
    expect(cancelled).toBe("ord-resting-1");
  });

  test("applies live market_data ticks from the WebSocket", async ({ page }) => {
    await page.route("**/api/auth/ws-token", (r) =>
      json(r, { token: "t", expires_in: 60, url: "ws://backend.test/api/v1/ws" }),
    );
    let resolveOpened: (v: { url: string; send: (m: string) => void }) => void;
    const opened = new Promise<{ url: string; send: (m: string) => void }>((res) => {
      resolveOpened = res;
    });
    await page.routeWebSocket(/backend\.test\/api\/v1\/ws/, (ws) => {
      ws.onMessage(() => {});
      resolveOpened({ url: ws.url(), send: (m) => ws.send(m) });
    });

    await page.goto("/trading");
    const ws = await opened;
    expect(new URL(ws.url).searchParams.get("channels")?.split(",")).toContain("market_data");
    await expect(page.getByTestId("last-day_ahead")).toHaveText("55.60");

    for (const price of [56.25, 57.5]) {
      ws.send(
        JSON.stringify({
          channel: "market_data",
          data: {
            event_id: `e${price}`,
            event_type: "market_data",
            source: "trading.simulation",
            severity: "info",
            timestamp: Date.now() / 1000,
            data: market("day_ahead", price),
          },
        }),
      );
    }
    await expect(page.getByTestId("last-day_ahead")).toHaveText("57.50");
    await expect(page.getByRole("img", { name: /day_ahead last price, 2 live ticks/ })).toBeVisible();
  });
});

test.describe("trading workspace (viewer)", () => {
  test.beforeEach(async ({ context }) => {
    await signInAs(context, "viewer");
    await stubTrading(context);
  });

  test("hides the order ticket and cancel actions", async ({ page }) => {
    await page.goto("/trading");
    await expect(page.getByTestId("read-only-note")).toContainText("viewer");
    await expect(page.getByTestId("market-rows")).toBeVisible();
    await expect(page.getByTestId("open-orders")).toBeVisible();
    await expect(page.getByTestId("order-ticket")).toHaveCount(0);
    await expect(page.getByRole("button", { name: /^Cancel/ })).toHaveCount(0);
  });
});

test.describe("strategies", () => {
  test.beforeEach(async ({ context }) => {
    await signInAs(context, "viewer");
    await stubTrading(context);
  });

  test("backtests a strategy and plots the equity curve", async ({ page }) => {
    let sent: Record<string, unknown> | null = null;
    await page.route(api("/api/v1/trading/strategies/momentum/backtest"), (r) => {
      sent = r.request().postDataJSON();
      return json(r, BACKTEST);
    });
    await page.goto("/trading/strategies");
    await expect(page.getByRole("link", { name: "Strategies" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    await page.getByLabel("momentum_threshold").fill("0.1");
    await page.getByTestId("bt-periods").fill("48");
    await page.getByTestId("bt-run").click();
    await expect(page.getByTestId("bt-total-pnl")).toContainText("$250.00");
    await expect(page.getByTestId("equity-curve")).toBeVisible();
    expect(sent).toMatchObject({
      params: { lookback_hours: 4, momentum_threshold: 0.1, base_quantity: 10 },
      synthetic: { periods: 48, interval_minutes: 60, seed: 42 },
    });
  });

  test("shows the server's reason when a backtest is rejected", async ({ page }) => {
    await page.route(api("/api/v1/trading/strategies/momentum/backtest"), (r) =>
      json(
        r,
        {
          detail: {
            code: "strategy_invalid",
            message: "lookback_hours must be positive",
            reasons: ["lookback_hours must be positive"],
          },
        },
        422,
      ),
    );
    await page.goto("/trading/strategies");
    await page.getByTestId("bt-run").click();
    await expect(page.getByTestId("error-state")).toContainText("lookback_hours must be positive");
  });
});
