"use client";

import Link from "next/link";
import type { Site } from "@/lib/api/types";
import { formatPower } from "@/lib/utils";

export function SitePopover({ site }: { site: Site }) {
  return (
    <div
      className="min-w-[14rem] space-y-1.5 p-2 text-sm"
      data-testid="site-popover"
    >
      <p className="font-semibold">{site.name}</p>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <dt className="text-muted-foreground">Resources</dt>
        <dd className="text-right tabular-nums">{site.total_resources}</dd>
        <dt className="text-muted-foreground">Online</dt>
        <dd className="text-right tabular-nums">
          {site.online_count} / {site.total_resources}
        </dd>
        <dt className="text-muted-foreground">Current power</dt>
        <dd className="text-right tabular-nums">
          {formatPower(site.current_power)}
        </dd>
        <dt className="text-muted-foreground">Active alerts</dt>
        <dd className="text-right tabular-nums">{site.active_alerts}</dd>
      </dl>
      <Link
        href={`/sites/${site.id}`}
        className="mt-1 inline-block text-xs font-medium text-primary hover:underline"
      >
        View site detail →
      </Link>
    </div>
  );
}
