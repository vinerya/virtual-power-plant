"use client";

import { Info } from "lucide-react";
import { apiErrorMessage, apiStatus } from "@/lib/api/errors";

/**
 * The bill endpoint answers 409 when no bill can be computed (no tariff
 * assigned, tariff deleted or unevaluable). That is an account state, not a
 * failure: show the server's explanation instead of a generic error.
 */
export function isBillUnavailable(err: unknown): boolean {
  return apiStatus(err) === 409;
}

export function BillUnavailable({ error }: { error: unknown }) {
  return (
    <div
      role="status"
      data-testid="bill-unavailable"
      className="flex items-start gap-3 rounded-md border bg-muted/40 p-4 text-sm"
    >
      <Info className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" aria-hidden="true" />
      <div className="space-y-1">
        <p className="font-medium">Your bill isn&apos;t available yet.</p>
        <p className="text-muted-foreground" data-testid="bill-unavailable-detail">
          {apiErrorMessage(error, "No bill can be computed for this account yet.")}
        </p>
        <p className="text-xs text-muted-foreground">
          Your energy provider sets this up; contact them if it doesn&apos;t appear soon.
        </p>
      </div>
    </div>
  );
}
