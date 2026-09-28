"use client";

import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/**
 * Inline error panel for failed queries. Announces itself to assistive tech
 * (role="alert"), shows the HTTP status when known, and offers a retry.
 */
export function ErrorState({
  title,
  error,
  onRetry,
  className,
  message,
}: {
  title: string;
  error?: unknown;
  onRetry?: () => void;
  className?: string;
  /** Server-provided explanation; replaces the generic HTTP-status text. */
  message?: string | null;
}) {
  const detail = message || describeError(error);
  return (
    <div
      role="alert"
      data-testid="error-state"
      className={cn(
        "flex flex-col items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm",
        className,
      )}
    >
      <p className="flex items-center gap-2 font-medium text-destructive">
        <AlertTriangle className="h-4 w-4" aria-hidden="true" />
        {title}
      </p>
      {detail && <p className="text-muted-foreground">{detail}</p>}
      {onRetry && (
        <Button type="button" variant="outline" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

function describeError(error: unknown): string | null {
  if (!error || typeof error !== "object") return null;
  const status = (error as { status?: number }).status;
  if (status === 404) {
    return "The backend does not provide this endpoint (404). Check that the API server is up to date.";
  }
  if (status === 401 || status === 403) {
    return "Your session is not authorized for this data. Try signing in again.";
  }
  if (status === 502 || status === 503 || status === 504) {
    return "The API server is unreachable.";
  }
  if (typeof status === "number") return `The API returned an error (HTTP ${status}).`;
  return "The request failed. Check your connection and the API server.";
}
