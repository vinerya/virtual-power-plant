"use client";

import { ChangePasswordCard, SessionsCard } from "@/components/settings/account-view";

export default function PortalAccountPage() {
  return (
    <div className="space-y-4" data-testid="portal-account">
      <h1 className="text-xl font-semibold tracking-tight">Account</h1>
      <div className="grid gap-4 md:grid-cols-2">
        <ChangePasswordCard />
        <SessionsCard />
      </div>
    </div>
  );
}
