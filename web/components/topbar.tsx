"use client";

import { Button } from "@/components/ui/button";
import { HealthPill } from "@/components/health-pill";
import { LiveStatusBadge } from "@/components/live-updates";

export function Topbar({ title }: { title: string }) {
  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/login";
  }
  return (
    <header className="flex h-14 items-center justify-between border-b bg-background px-4 md:px-6">
      <h1 className="text-base font-semibold tracking-tight">{title}</h1>
      <div className="flex items-center gap-3">
        <LiveStatusBadge />
        <HealthPill />
        <Button variant="outline" size="sm" onClick={logout}>
          Sign out
        </Button>
      </div>
    </header>
  );
}
