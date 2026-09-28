import { SettingsView } from "@/components/settings/settings-view";

export const metadata = {
  title: "Settings · VPP Console",
};

export default function SettingsPage() {
  return (
    <div className="space-y-3">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight">Settings</h2>
        <p className="text-sm text-muted-foreground">
          Edit the runtime YAML configuration. Changes are validated against
          the backend&apos;s JSON Schema and require explicit Apply.
        </p>
      </header>
      <SettingsView />
    </div>
  );
}
