"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getMe } from "@/lib/api/customer";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/portal", label: "Overview" },
  { href: "/portal/bill", label: "Bill" },
  { href: "/portal/devices", label: "Devices" },
  { href: "/portal/enroll", label: "Programs" },
  { href: "/portal/account", label: "Account" },
];

export default function CustomerLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const me = useQuery({ queryKey: ["customer", "me"], queryFn: getMe });

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/login";
  }

  return (
    <div className="min-h-screen bg-[hsl(210_40%_98%)] text-foreground">
      <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
          <Link href="/portal" className="flex items-center gap-2">
            <Sun className="h-5 w-5 text-amber-500" aria-hidden="true" />
            <span className="text-base font-semibold tracking-tight">
              VPP Member Portal
            </span>
          </Link>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-muted-foreground sm:inline">
              {me.data?.name ?? "Member"}
            </span>
            <Button variant="outline" size="sm" onClick={logout}>
              Sign out
            </Button>
          </div>
        </div>
        <nav
          aria-label="Customer portal"
          className="mx-auto flex max-w-5xl gap-2 overflow-x-auto px-4 pb-2"
        >
          {NAV.map((item) => {
            const active =
              pathname === item.href ||
              (item.href !== "/portal" &&
                pathname?.startsWith(item.href + "/"));
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "rounded-full px-3 py-1.5 text-sm transition-colors",
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-8 md:px-6 md:py-10">
        {children}
      </main>
    </div>
  );
}
