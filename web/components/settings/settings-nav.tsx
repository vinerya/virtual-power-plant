"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession } from "@/lib/api/session";
import { cn } from "@/lib/utils";

const ITEMS = [
  { href: "/settings", label: "Configuration", adminOnly: false },
  { href: "/settings/account", label: "Account", adminOnly: false },
  { href: "/settings/users", label: "Users & API keys", adminOnly: true },
] as const;

/** Sub-navigation shared by the settings pages. */
export function SettingsNav() {
  const pathname = usePathname();
  const session = useSession();
  const isAdmin = session.data?.role === "admin";
  return (
    <nav aria-label="Settings sections" className="flex gap-1 border-b" data-testid="settings-nav">
      {ITEMS.filter((i) => !i.adminOnly || isAdmin).map((item) => {
        const active = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors",
              active
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
