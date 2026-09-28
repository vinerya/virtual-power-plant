"use client";

import { Button } from "@/components/ui/button";
import { HealthPill } from "@/components/health-pill";
import { LiveStatusBadge } from "@/components/live-updates";
import { useSession } from "@/lib/api/session";

function SessionChip() {
  const q = useSession();
  if (!q.data?.authenticated || !q.data.username) return null;
  return (
    <span
      className="hidden text-xs text-muted-foreground sm:inline"
      data-testid="session-chip"
      title="Signed-in user and role"
    >
      {q.data.username}
      {q.data.role ? ` · ${q.data.role}` : ""}
    </span>
  );
}

export function Topbar({ title }: { title: string }) {
  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/login";
  }
  return (
    <header className="flex h-14 items-center justify-between border-b bg-background px-4 md:px-6">
      <h1 className="text-base font-semibold tracking-tight">{title}</h1>
      <div className="flex items-center gap-3">
        <SessionChip />
        <LiveStatusBadge />
        <HealthPill />
        <Button variant="outline" size="sm" onClick={logout}>
          Sign out
        </Button>
      </div>
    </header>
  );
}
