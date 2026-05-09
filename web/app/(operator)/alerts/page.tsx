import { AlertsList } from "@/components/alerts/alerts-list";

export const metadata = { title: "Alerts · VPP Operator Console" };

export default function AlertsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Alerts</h2>
        <p className="text-sm text-muted-foreground">
          Live-tailed events from across the fleet.
        </p>
      </div>
      <AlertsList />
    </div>
  );
}
