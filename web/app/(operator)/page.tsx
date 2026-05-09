import { FleetOverview } from "@/components/fleet-overview";

export default function FleetPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Fleet overview</h2>
        <p className="text-sm text-muted-foreground">
          Live snapshot of every registered energy resource.
        </p>
      </div>
      <FleetOverview />
    </div>
  );
}
