"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Check } from "lucide-react";
import { SnoozePicker } from "./snooze-picker";
import type { Alert } from "@/lib/api/types";
import { cn, formatDateTime, formatRelativeTime } from "@/lib/utils";

export function AlertRow({
  alert,
  selected,
  onToggleSelect,
  onAck,
  onSnooze,
  pulseUntil,
}: {
  alert: Alert;
  selected: boolean;
  onToggleSelect: () => void;
  onAck: () => void;
  onSnooze: (durationMs: number) => void;
  pulseUntil?: number;
}) {
  const [pulsing, setPulsing] = useState(false);
  const ref = useRef<HTMLLIElement>(null);

  useEffect(() => {
    if (!pulseUntil) return;
    const remaining = pulseUntil - Date.now();
    if (remaining <= 0) return;
    setPulsing(true);
    const t = setTimeout(() => setPulsing(false), remaining);
    return () => clearTimeout(t);
  }, [pulseUntil]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === " ") {
      e.preventDefault();
      onToggleSelect();
    } else if (e.key === "a" || e.key === "A") {
      onAck();
    }
  };

  return (
    <li
      ref={ref}
      tabIndex={0}
      role="row"
      aria-selected={selected}
      onKeyDown={onKeyDown}
      data-testid="alert-row"
      data-severity={alert.severity}
      data-status={alert.status}
      className={cn(
        "flex flex-wrap items-start gap-3 border-b px-3 py-3 outline-none last:border-b-0 hover:bg-muted/30 focus-visible:ring-2 focus-visible:ring-ring",
        pulsing && "animate-pulse bg-primary/10",
        alert.status !== "active" && "opacity-70",
      )}
    >
      <input
        type="checkbox"
        aria-label={`Select alert ${alert.title}`}
        checked={selected}
        onChange={onToggleSelect}
        className="mt-1"
        data-testid="alert-checkbox"
      />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <SeverityBadge severity={alert.severity} />
          <span className="text-sm font-medium" data-testid="alert-title">
            {alert.title}
          </span>
          <StatusPill status={alert.status} />
          <span
            className="text-xs text-muted-foreground"
            title={formatDateTime(alert.timestamp)}
          >
            {formatRelativeTime(alert.timestamp)}
          </span>
        </div>
        <p className="mt-0.5 text-sm text-muted-foreground">{alert.message}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Source:{" "}
          <code className="font-mono">{alert.source}</code>
        </p>
      </div>
      <div className="flex flex-shrink-0 items-center gap-2">
        {alert.source_link && (
          <Button asChild variant="outline" size="sm">
            <Link href={alert.source_link} aria-label="Open source">
              <ExternalLink className="mr-1 h-3.5 w-3.5" /> Open
            </Link>
          </Button>
        )}
        {alert.status === "active" && (
          <>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onAck}
              data-testid="ack-button"
            >
              <Check className="mr-1 h-3.5 w-3.5" /> Ack
            </Button>
            <SnoozePicker onPick={onSnooze} />
          </>
        )}
      </div>
    </li>
  );
}

function SeverityBadge({ severity }: { severity: Alert["severity"] }) {
  switch (severity) {
    case "critical":
      return <Badge variant="destructive">critical</Badge>;
    case "warning":
      return (
        <Badge
          variant="outline"
          className="border-amber-500 text-amber-600 dark:text-amber-400"
        >
          warning
        </Badge>
      );
    default:
      return <Badge variant="secondary">info</Badge>;
  }
}

function StatusPill({ status }: { status: Alert["status"] }) {
  if (status === "active") return null;
  return (
    <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
      {status}
    </Badge>
  );
}
