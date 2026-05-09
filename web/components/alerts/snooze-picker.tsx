"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Clock } from "lucide-react";
import { cn } from "@/lib/utils";

const OPTIONS: { label: string; ms: number }[] = [
  { label: "15m", ms: 15 * 60 * 1000 },
  { label: "1h", ms: 60 * 60 * 1000 },
  { label: "4h", ms: 4 * 60 * 60 * 1000 },
  { label: "24h", ms: 24 * 60 * 60 * 1000 },
];

export function SnoozePicker({
  onPick,
  disabled,
  size = "sm",
}: {
  onPick: (durationMs: number) => void;
  disabled?: boolean;
  size?: "sm" | "default";
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative inline-block" data-testid="snooze-picker">
      <Button
        type="button"
        variant="outline"
        size={size}
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <Clock className="mr-1 h-3.5 w-3.5" /> Snooze
      </Button>
      {open && (
        <div
          role="menu"
          className={cn(
            "absolute right-0 z-20 mt-1 w-32 rounded-md border bg-popover p-1 shadow-md",
          )}
          onMouseLeave={() => setOpen(false)}
        >
          {OPTIONS.map((o) => (
            <button
              key={o.label}
              role="menuitem"
              type="button"
              className="flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-sm hover:bg-accent"
              onClick={() => {
                onPick(o.ms);
                setOpen(false);
              }}
              data-testid={`snooze-${o.label}`}
            >
              <span>{o.label}</span>
              <span className="text-xs text-muted-foreground">
                {humanize(o.ms)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function humanize(ms: number): string {
  const min = ms / 60000;
  if (min < 60) return `${min}m`;
  return `${min / 60}h`;
}
