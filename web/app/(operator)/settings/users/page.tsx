import { SettingsNav } from "@/components/settings/settings-nav";
import { UsersView } from "@/components/settings/users-view";

export const metadata = {
  title: "Users & API keys · VPP Console",
};

export default function UsersPage() {
  return (
    <div className="space-y-3">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight">Settings</h2>
        <p className="text-sm text-muted-foreground">
          Manage user accounts, roles and every API key (admin).
        </p>
      </header>
      <SettingsNav />
      <UsersView />
    </div>
  );
}
