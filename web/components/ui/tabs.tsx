"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface TabsCtx {
  value: string;
  setValue: (v: string) => void;
  baseId: string;
}
const Ctx = React.createContext<TabsCtx | null>(null);

const safe = (v: string) => v.replace(/[^a-zA-Z0-9_-]/g, "_");

export function Tabs({
  value,
  defaultValue,
  onValueChange,
  className,
  children,
}: {
  value?: string;
  defaultValue?: string;
  onValueChange?: (v: string) => void;
  className?: string;
  children: React.ReactNode;
}) {
  const [internal, setInternal] = React.useState(defaultValue ?? "");
  const baseId = React.useId();
  const v = value ?? internal;
  const setValue = React.useCallback(
    (next: string) => {
      if (value === undefined) setInternal(next);
      onValueChange?.(next);
    },
    [value, onValueChange],
  );
  return (
    <Ctx.Provider value={{ value: v, setValue, baseId }}>
      <div className={cn(className)}>{children}</div>
    </Ctx.Provider>
  );
}

/**
 * WAI-ARIA tabs keyboard model: Left/Right (and Home/End) move between
 * tabs and activate them; only the active tab is in the tab order.
 */
function onTabListKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
  const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
  if (!keys.includes(e.key)) return;
  const tabs = Array.from(
    e.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]:not([disabled])'),
  );
  const idx = tabs.findIndex((t) => t === document.activeElement);
  if (idx === -1 || tabs.length === 0) return;
  e.preventDefault();
  const next =
    e.key === "Home"
      ? 0
      : e.key === "End"
        ? tabs.length - 1
        : (idx + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus();
  tabs[next].click();
}

export function TabsList({
  className,
  onKeyDown,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      role="tablist"
      onKeyDown={(e) => {
        onTabListKeyDown(e);
        onKeyDown?.(e);
      }}
      className={cn(
        "inline-flex h-9 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground",
        className,
      )}
      {...props}
    />
  );
}

export function TabsTrigger({
  value,
  className,
  children,
  ...props
}: { value: string } & Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  "value"
>) {
  const ctx = React.useContext(Ctx);
  if (!ctx) throw new Error("TabsTrigger must be inside Tabs");
  const active = ctx.value === value;
  return (
    <button
      type="button"
      role="tab"
      id={`${ctx.baseId}-tab-${safe(value)}`}
      aria-controls={`${ctx.baseId}-panel-${safe(value)}`}
      aria-selected={active}
      tabIndex={active ? 0 : -1}
      data-state={active ? "active" : "inactive"}
      onClick={() => ctx.setValue(value)}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        active
          ? "bg-background text-foreground shadow"
          : "hover:text-foreground",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function TabsContent({
  value,
  className,
  ...props
}: { value: string } & React.HTMLAttributes<HTMLDivElement>) {
  const ctx = React.useContext(Ctx);
  if (!ctx) throw new Error("TabsContent must be inside Tabs");
  if (ctx.value !== value) return null;
  return (
    <div
      role="tabpanel"
      id={`${ctx.baseId}-panel-${safe(value)}`}
      aria-labelledby={`${ctx.baseId}-tab-${safe(value)}`}
      tabIndex={0}
      className={cn(
        "mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        className,
      )}
      {...props}
    />
  );
}
