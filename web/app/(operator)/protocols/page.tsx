import { ProtocolsView } from "@/components/protocols/protocols-view";

export default function ProtocolsPage() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Protocols</h2>
        <p className="text-sm text-muted-foreground">
          Device and grid protocol adapters, whether each talks to real equipment (live) or
          runs in memory only (simulated), and their traffic counters.
        </p>
      </div>
      <ProtocolsView />
    </div>
  );
}
