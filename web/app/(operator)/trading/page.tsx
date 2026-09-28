import { TradingNav } from "@/components/trading/trading-nav";
import { TradingWorkspace } from "@/components/trading/trading-workspace";

export default function TradingPage() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Trading</h2>
        <p className="text-sm text-muted-foreground">
          Markets, order entry, open orders, fills and portfolio risk.
        </p>
      </div>
      <TradingNav />
      <TradingWorkspace />
    </div>
  );
}
