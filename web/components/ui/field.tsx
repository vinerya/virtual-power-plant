"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Label + control + optional hint/error, wired for assistive tech: the
 * label targets the control by id and hint/error are referenced through
 * aria-describedby (the error also sets aria-invalid).
 */
export function Field({
  label,
  hint,
  error,
  className,
  children,
}: {
  label: React.ReactNode;
  hint?: React.ReactNode;
  error?: string | null;
  className?: string;
  children: React.ReactElement<React.InputHTMLAttributes<HTMLElement>>;
}) {
  const auto = React.useId();
  const id = (children.props.id as string | undefined) ?? auto;
  const hintId = hint ? `${id}-hint` : undefined;
  const errId = error ? `${id}-err` : undefined;
  const describedBy = [hintId, errId].filter(Boolean).join(" ") || undefined;
  return (
    <div className={cn("space-y-1.5", className)}>
      <label htmlFor={id} className="text-xs font-medium text-muted-foreground">
        {label}
      </label>
      {React.cloneElement(children, {
        id,
        "aria-describedby": describedBy,
        "aria-invalid": error ? true : undefined,
      })}
      {hint && (
        <p id={hintId} className="text-xs text-muted-foreground">
          {hint}
        </p>
      )}
      {error && (
        <p id={errId} className="text-xs text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}

/** Compact label/value pair for stat grids. */
export function Stat({
  label,
  value,
  sub,
  tone,
  testId,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "positive" | "negative" | "warning";
  testId?: string;
}) {
  return (
    <div className="rounded-md border bg-background px-3 py-2" data-testid={testId}>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd
        className={cn(
          "text-base font-semibold tabular-nums",
          tone === "positive" && "text-emerald-600",
          tone === "negative" && "text-destructive",
          tone === "warning" && "text-amber-600",
        )}
      >
        {value}
      </dd>
      {sub && <dd className="text-[11px] text-muted-foreground">{sub}</dd>}
    </div>
  );
}

export function pnlTone(v: number | null | undefined): "positive" | "negative" | undefined {
  if (v == null || !Number.isFinite(v) || Math.abs(v) < 1e-9) return undefined;
  return v > 0 ? "positive" : "negative";
}
