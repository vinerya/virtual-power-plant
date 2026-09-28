import { AuditView } from "@/components/settings/audit-view";
import { SettingsNav } from "@/components/settings/settings-nav";

export const metadata = {
  title: "Audit log · VPP Console",
};

export default function AuditPage() {
  return (
    <div className="space-y-3">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight">Settings</h2>
        <p className="text-sm text-muted-foreground">
          Who signed in, changed accounts or keys, and sent control actions (admin).
        </p>
      </header>
      <SettingsNav />
      <AuditView />
    </div>
  );
}
