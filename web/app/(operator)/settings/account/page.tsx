import { AccountView } from "@/components/settings/account-view";
import { SettingsNav } from "@/components/settings/settings-nav";

export const metadata = {
  title: "Account · VPP Console",
};

export default function AccountPage() {
  return (
    <div className="space-y-3">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight">Settings</h2>
        <p className="text-sm text-muted-foreground">Your password, sessions and API keys.</p>
      </header>
      <SettingsNav />
      <AccountView />
    </div>
  );
}
