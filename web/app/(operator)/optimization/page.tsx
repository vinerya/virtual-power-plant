"use client";

import { useState } from "react";
import { BacktestPanel } from "@/components/optimization/backtest-panel";
import { SchedulePlanner } from "@/components/optimization/schedule-planner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function OptimizationPage() {
  const [tab, setTab] = useState("schedule");
  return (
    <div className="space-y-4" data-testid="optimization-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Optimization planner</h2>
        <p className="text-sm text-muted-foreground">
          Plan battery schedules over a horizon and measure the controller against baselines.
          Every run is saved and can be opened in the dispatch explainer.
        </p>
      </div>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList aria-label="Planner mode">
          <TabsTrigger value="schedule" data-testid="tab-schedule">
            Horizon schedule
          </TabsTrigger>
          <TabsTrigger value="backtest" data-testid="tab-backtest">
            Backtest vs. baselines
          </TabsTrigger>
        </TabsList>
        <TabsContent value="schedule">
          <SchedulePlanner />
        </TabsContent>
        <TabsContent value="backtest">
          <BacktestPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
