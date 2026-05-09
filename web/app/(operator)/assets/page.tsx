import { FleetOverview } from "@/components/fleet-overview";

export default function AssetsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Assets</h2>
        <p className="text-sm text-muted-foreground">
          All registered energy resources. Click an asset to inspect its live
          telemetry and subtype-specific metrics.
        </p>
      </div>
      <FleetOverview />
    </div>
  );
}
