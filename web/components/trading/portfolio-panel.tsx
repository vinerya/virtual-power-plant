"use client";

import { AlertTriangle, ShieldCheck } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Stat, pnlTone } from "@/components/ui/field";
import { Skeleton } from "@/components/ui/skeleton";
import type { Portfolio } from "@/lib/api/trading";
import {
  cn,
  formatMoney,
  formatNumber,
  formatPercent,
  formatRelativeTime,
  parseApiDate,
} from "@/lib/utils";

export function PortfolioPanel({
  portfolio,
  isLoading,
  error,
  onRetry,
}: {
  portfolio: Portfolio | undefined;
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
}) {
  return (
    <Card data-testid="portfolio-panel">
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-base text-foreground">Portfolio &amp; risk</CardTitle>
        {portfolio && (
          <span className="text-xs text-muted-foreground">
            updated {formatRelativeTime(parseApiDate(portfolio.last_updated))}
          </span>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <Skeleton className="h-40 w-full" />
        ) : error || !portfolio ? (
          <ErrorState title="Failed to load portfolio." error={error} onRetry={onRetry} />
        ) : (
          <PortfolioBody p={portfolio} />
        )}
      </CardContent>
    </Card>
  );
}

function PortfolioBody({ p }: { p: Portfolio }) {
  const risk = p.risk;
  const limits = risk?.limits;
  const breaches = risk?.breaches ?? [];
  return (
    <>
      {breaches.length > 0 ? (
        <div
          role="alert"
          data-testid="limit-breaches"
          className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm"
        >
          <p className="flex items-center gap-2 font-medium text-destructive">
            <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            {breaches.length} risk limit breach{breaches.length === 1 ? "" : "es"}
          </p>
          <ul className="mt-1 list-disc pl-6 text-xs">
            {breaches.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="flex items-center gap-2 text-xs text-muted-foreground" data-testid="no-breaches">
          <ShieldCheck className="h-4 w-4 text-emerald-600" aria-hidden="true" />
          All risk limits within bounds.
        </p>
      )}

      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 2xl:grid-cols-5">
        <Stat label="Equity" value={formatMoney(p.equity)} sub={`cash ${formatMoney(p.cash)}`} />
        <Stat
          label="Total P&L"
          value={formatMoney(p.total_pnl)}
          tone={pnlTone(p.total_pnl)}
          sub={`fees ${formatMoney(p.fees_paid)}`}
          testId="stat-total-pnl"
        />
        <Stat
          label="Realized P&L"
          value={formatMoney(p.realized_pnl)}
          tone={pnlTone(p.realized_pnl)}
        />
        <Stat
          label="Unrealized P&L"
          value={formatMoney(p.unrealized_pnl)}
          tone={pnlTone(p.unrealized_pnl)}
        />
        <Stat
          label="Gross / net exposure"
          value={formatMoney(p.gross_exposure)}
          sub={`net ${formatMoney(p.net_exposure)}`}
        />
        <Stat
          label="VaR 95% 1-day"
          value={formatMoney(risk?.var_95_1d)}
          sub={limits ? `limit ${formatMoney(limits.var_limit)}` : undefined}
          tone={
            risk?.var_95_1d != null && limits && risk.var_95_1d > limits.var_limit
              ? "negative"
              : undefined
          }
          testId="stat-var"
        />
        <Stat
          label="Drawdown (current / max)"
          value={`${formatPercent(p.current_drawdown)} / ${formatPercent(p.max_drawdown)}`}
          sub={limits ? `limit ${formatPercent(limits.max_drawdown)}` : undefined}
          tone={limits && p.current_drawdown > limits.max_drawdown ? "negative" : undefined}
        />
        <Stat
          label="Daily P&L"
          value={formatMoney(risk?.daily_pnl)}
          tone={pnlTone(risk?.daily_pnl)}
          sub={limits ? `loss limit ${formatMoney(limits.max_daily_loss)}` : undefined}
        />
        <Stat label="Open orders / trades" value={`${p.open_orders} / ${p.total_trades}`} />
      </dl>

      <section aria-labelledby="positions-h">
        <h3 id="positions-h" className="mb-1 text-sm font-semibold">
          Positions
        </h3>
        {p.positions.length === 0 ? (
          <p className="text-sm text-muted-foreground">Flat: no open positions.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="positions-table">
              <thead className="border-b text-xs uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th scope="col" className="py-2 pr-3 text-left font-medium">Market</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Qty</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Avg</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Mark</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Unrealized</th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">Realized</th>
                  <th scope="col" className="py-2 text-right font-medium">Concentration</th>
                </tr>
              </thead>
              <tbody>
                {p.positions.map((pos) => {
                  const conc = risk?.concentrations?.[pos.market];
                  const overLimit =
                    conc != null && limits != null && conc > limits.concentration_limit;
                  const overPos =
                    limits != null && Math.abs(pos.quantity) > limits.max_position;
                  return (
                    <tr key={pos.market} className="border-b last:border-b-0">
                      <th scope="row" className="py-2 pr-3 text-left font-medium">
                        {pos.market}
                      </th>
                      <td
                        className={cn(
                          "py-2 pr-3 text-right tabular-nums",
                          overPos && "text-destructive",
                        )}
                      >
                        {formatNumber(pos.quantity, 1)}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums">
                        {formatNumber(pos.average_price)}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums">
                        {formatNumber(pos.mark_price)}
                      </td>
                      <td
                        className={cn(
                          "py-2 pr-3 text-right tabular-nums",
                          pos.unrealized_pnl > 0 && "text-emerald-600",
                          pos.unrealized_pnl < 0 && "text-destructive",
                        )}
                      >
                        {formatNumber(pos.unrealized_pnl)}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums">
                        {formatNumber(pos.realized_pnl)}
                      </td>
                      <td
                        className={cn(
                          "py-2 text-right tabular-nums",
                          overLimit && "text-destructive",
                        )}
                      >
                        {conc != null ? formatPercent(conc) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {risk && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer select-none rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            Risk method &amp; limits
          </summary>
          <div className="mt-2 space-y-1">
            <p>VaR: {risk.var_method || "not reported"}.</p>
            {Object.entries(risk.volatility).map(([m, v]) => (
              <p key={m}>
                {m} volatility {formatPercent(v.value)} ({v.source}, {v.samples} samples)
              </p>
            ))}
            {limits && (
              <p>
                Limits: position {limits.max_position}, daily loss{" "}
                {formatMoney(limits.max_daily_loss)}, drawdown {formatPercent(limits.max_drawdown)},
                VaR {formatMoney(limits.var_limit)}, concentration{" "}
                {formatPercent(limits.concentration_limit)}.
              </p>
            )}
          </div>
        </details>
      )}
    </>
  );
}
