import { StrategiesView } from "@/components/trading/strategies-view";
import { TradingNav } from "@/components/trading/trading-nav";

export default function StrategiesPage() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Trading strategies</h2>
        <p className="text-sm text-muted-foreground">
          Built-in strategies and their default parameters. Backtest any of them over seeded
          synthetic prices.
        </p>
      </div>
      <TradingNav />
      <StrategiesView />
    </div>
  );
}
