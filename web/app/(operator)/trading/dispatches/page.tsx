import { DispatchesView } from "@/components/dispatch/dispatches-view";

export default function DispatchesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Dispatch history</h2>
        <p className="text-sm text-muted-foreground">
          Past optimization runs. Click any row for inputs, schedule, and
          diagnostics.
        </p>
      </div>
      <DispatchesView />
    </div>
  );
}
