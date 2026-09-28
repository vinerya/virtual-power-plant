"use client";

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { AlertTriangle, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import {
  ORDER_TYPES,
  TIME_IN_FORCE,
  submitOrder,
  tradingError,
  type Market,
  type OrderCreate,
  type OrderType,
} from "@/lib/api/trading";
import { apiErrorMessage, apiStatus } from "@/lib/api/errors";
import { cn, formatNumber } from "@/lib/utils";
import { TRADING_KEYS } from "./use-market-stream";

const TYPE_LABEL: Record<OrderType, string> = {
  market: "Market",
  limit: "Limit",
  stop: "Stop",
  stop_limit: "Stop-limit",
  iceberg: "Iceberg",
  ioc: "Immediate-or-cancel",
  fok: "Fill-or-kill",
};

const needsPrice = (t: OrderType) => ["limit", "iceberg", "ioc", "fok"].includes(t);
const needsStop = (t: OrderType) => t === "stop" || t === "stop_limit";
const hasTif = (t: OrderType) => ["limit", "stop", "stop_limit", "iceberg"].includes(t);

interface Rejection {
  title: string;
  reasons: string[];
  orderId?: string;
}

export function OrderTicket({
  markets,
  market,
  onMarketChange,
}: {
  markets: Market[];
  market: string | undefined;
  onMarketChange: (m: string) => void;
}) {
  const qc = useQueryClient();
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [type, setType] = useState<OrderType>("limit");
  const [quantity, setQuantity] = useState("1");
  const [price, setPrice] = useState("");
  const [stopPrice, setStopPrice] = useState("");
  const [limitPrice, setLimitPrice] = useState("");
  const [visible, setVisible] = useState("");
  const [tif, setTif] = useState<(typeof TIME_IN_FORCE)[number]>("GTC");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [rejection, setRejection] = useState<Rejection | null>(null);

  const m = markets.find((x) => x.market === market);

  // Prefill a sensible limit price from the book when switching market/side.
  useEffect(() => {
    if (!m) return;
    const ref = side === "buy" ? (m.bid ?? m.last_price) : (m.ask ?? m.last_price);
    if (ref != null) setPrice(ref.toFixed(2));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [market, side]);

  const mutation = useMutation({
    mutationFn: submitOrder,
    onSuccess: (res) => {
      setRejection(null);
      const filled = res.fills.reduce((s, f) => s + f.quantity, 0);
      toast.success(`Order ${res.status}`, {
        description:
          res.fills.length > 0
            ? `${formatNumber(filled, 1)} ${m?.quantity_unit ?? "MWh"} filled at avg ${formatNumber(res.average_price)}`
            : `${res.side} ${res.quantity} ${res.market} — ${res.status}`,
      });
      void qc.invalidateQueries({ queryKey: TRADING_KEYS.orders });
      void qc.invalidateQueries({ queryKey: TRADING_KEYS.trades });
      void qc.invalidateQueries({ queryKey: TRADING_KEYS.portfolio });
    },
    onError: (err) => {
      const t = tradingError(err);
      if (t) {
        setRejection({
          title:
            t.code === "risk_limit_breached"
              ? "Rejected by pre-trade risk checks"
              : t.code === "unknown_market"
                ? "Unknown market"
                : "Order rejected",
          reasons: t.reasons.length ? t.reasons : [t.message],
          orderId: t.order_id,
        });
        if (t.code === "risk_limit_breached") {
          // The rejected order is persisted for the audit trail.
          void qc.invalidateQueries({ queryKey: TRADING_KEYS.orders });
        }
        return;
      }
      const status = apiStatus(err);
      setRejection({
        title: status === 403 ? "Not permitted" : "Order failed",
        reasons: [
          status === 403
            ? "Your role cannot place orders (operator or admin required)."
            : apiErrorMessage(err, "The order could not be submitted."),
        ],
      });
    },
  });

  function validate(): OrderCreate | null {
    const e: Record<string, string> = {};
    const qty = Number(quantity);
    if (!market) e.market = "Choose a market";
    if (!(qty > 0)) e.quantity = "Quantity must be greater than 0";
    const num = (s: string) => (s.trim() === "" ? NaN : Number(s));
    const p = num(price);
    const sp = num(stopPrice);
    const lp = num(limitPrice);
    const vis = num(visible);
    if (needsPrice(type) && !(p > 0)) e.price = "Enter a positive limit price";
    if (needsStop(type) && !(sp > 0)) e.stop_price = "Enter a positive stop trigger price";
    if (type === "stop_limit" && !(lp > 0)) e.limit_price = "Enter a positive limit price";
    if (type === "iceberg" && !(vis > 0 && vis <= qty))
      e.visible_quantity = "Visible quantity must be > 0 and ≤ quantity";
    setErrors(e);
    if (Object.keys(e).length) return null;
    const body: OrderCreate = { order_type: type, market: market!, side, quantity: qty };
    if (needsPrice(type)) body.price = p;
    if (needsStop(type)) body.stop_price = sp;
    if (type === "stop_limit") body.limit_price = lp;
    if (type === "iceberg") body.visible_quantity = vis;
    if (hasTif(type)) body.time_in_force = tif;
    return body;
  }

  function onSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    setRejection(null);
    const body = validate();
    if (body) mutation.mutate(body);
  }

  return (
    <Card data-testid="order-ticket">
      <CardHeader className="pb-3">
        <CardTitle className="text-base text-foreground">Order ticket</CardTitle>
        <p className="text-xs text-muted-foreground">
          Orders are matched by the simulated venue and pass the same pre-trade risk
          checks as live orders would.
        </p>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-3" noValidate aria-label="Order ticket">
          <fieldset>
            <legend className="mb-1.5 text-xs font-medium text-muted-foreground">Side</legend>
            <div className="grid grid-cols-2 gap-2">
              {(["buy", "sell"] as const).map((s) => (
                <label
                  key={s}
                  className={cn(
                    "flex cursor-pointer items-center justify-center rounded-md border px-3 py-2 text-sm font-semibold capitalize focus-within:ring-2 focus-within:ring-ring",
                    side === s
                      ? s === "buy"
                        ? "border-emerald-500 bg-emerald-50 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
                        : "border-red-500 bg-red-50 text-red-800 dark:bg-red-950 dark:text-red-200"
                      : "text-muted-foreground",
                  )}
                >
                  <input
                    type="radio"
                    name="side"
                    value={s}
                    checked={side === s}
                    onChange={() => setSide(s)}
                    className="sr-only"
                  />
                  {s}
                </label>
              ))}
            </div>
          </fieldset>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Market" error={errors.market}>
              <Select
                value={market ?? ""}
                onChange={(e) => onMarketChange(e.target.value)}
                data-testid="ticket-market"
              >
                {markets.map((x) => (
                  <option key={x.market} value={x.market}>
                    {x.market}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Order type">
              <Select
                value={type}
                onChange={(e) => setType(e.target.value as OrderType)}
                data-testid="ticket-type"
              >
                {ORDER_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {TYPE_LABEL[t]}
                  </option>
                ))}
              </Select>
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field
              label={`Quantity (${m?.quantity_unit ?? "MWh"})`}
              error={errors.quantity}
              hint={m ? `Lot size ${m.lot_size}` : undefined}
            >
              <Input
                type="number"
                inputMode="decimal"
                min={0}
                step={m?.lot_size ?? 0.1}
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                data-testid="ticket-quantity"
              />
            </Field>
            {needsPrice(type) && (
              <Field
                label={`Limit price (${m?.price_unit ?? "$/MWh"})`}
                error={errors.price}
              >
                <Input
                  type="number"
                  inputMode="decimal"
                  min={0}
                  step={m?.tick_size ?? 0.01}
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  data-testid="ticket-price"
                />
              </Field>
            )}
            {needsStop(type) && (
              <Field label="Stop trigger" error={errors.stop_price}>
                <Input
                  type="number"
                  inputMode="decimal"
                  min={0}
                  step={m?.tick_size ?? 0.01}
                  value={stopPrice}
                  onChange={(e) => setStopPrice(e.target.value)}
                />
              </Field>
            )}
            {type === "stop_limit" && (
              <Field label="Limit after trigger" error={errors.limit_price}>
                <Input
                  type="number"
                  inputMode="decimal"
                  min={0}
                  step={m?.tick_size ?? 0.01}
                  value={limitPrice}
                  onChange={(e) => setLimitPrice(e.target.value)}
                />
              </Field>
            )}
            {type === "iceberg" && (
              <Field label="Visible quantity" error={errors.visible_quantity}>
                <Input
                  type="number"
                  inputMode="decimal"
                  min={0}
                  value={visible}
                  onChange={(e) => setVisible(e.target.value)}
                />
              </Field>
            )}
            {hasTif(type) && (
              <Field label="Time in force">
                <Select
                  value={tif}
                  onChange={(e) => setTif(e.target.value as (typeof TIME_IN_FORCE)[number])}
                >
                  {TIME_IN_FORCE.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </Select>
              </Field>
            )}
          </div>

          {m && (
            <p className="text-xs text-muted-foreground">
              Fee {formatNumber(m.fee_per_unit)} per {m.quantity_unit}. Bid{" "}
              {formatNumber(m.bid)} / Ask {formatNumber(m.ask)}.
            </p>
          )}

          {rejection && (
            <div
              role="alert"
              data-testid="order-rejection"
              className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm"
            >
              <p className="flex items-center gap-2 font-medium text-destructive">
                <AlertTriangle className="h-4 w-4" aria-hidden="true" />
                {rejection.title}
              </p>
              <ul className="mt-1 list-disc space-y-0.5 pl-6 text-xs">
                {rejection.reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
              {rejection.orderId && (
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Recorded as rejected order <code>{rejection.orderId.slice(0, 8)}</code> for the
                  audit trail.
                </p>
              )}
            </div>
          )}

          <Button
            type="submit"
            className={cn(
              "w-full",
              side === "sell" && "bg-red-600 text-white hover:bg-red-600/90",
              side === "buy" && "bg-emerald-600 text-white hover:bg-emerald-600/90",
            )}
            disabled={mutation.isPending || !markets.length}
            data-testid="ticket-submit"
          >
            <Send className="h-4 w-4" aria-hidden="true" />
            {mutation.isPending
              ? "Submitting…"
              : `${side === "buy" ? "Buy" : "Sell"} ${quantity || 0} ${market ?? ""}`}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
