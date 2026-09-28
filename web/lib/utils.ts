import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPower(kw: number): string {
  if (!Number.isFinite(kw)) return "—";
  const abs = Math.abs(kw);
  if (abs >= 1000) return `${(kw / 1000).toFixed(2)} MW`;
  return `${kw.toFixed(1)} kW`;
}

export function formatPercent(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export function formatDateTime(iso: string | Date | null | undefined): string {
  if (!iso) return "—";
  const d = typeof iso === "string" ? new Date(iso) : iso;
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString();
}

export function formatRelativeTime(
  iso: string | Date | null | undefined,
): string {
  if (!iso) return "—";
  const d = typeof iso === "string" ? new Date(iso) : iso;
  if (Number.isNaN(d.getTime())) return "—";
  const diff = Date.now() - d.getTime();
  const sec = Math.round(diff / 1000);
  if (Math.abs(sec) < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (Math.abs(min) < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (Math.abs(hr) < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  return `${day}d ago`;
}

/** `$1,234.56` style money (sign preserved). `—` for non-finite values. */
export function formatMoney(v: number | null | undefined, currency = "USD"): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(v);
}

/** Fixed-decimals number with thousands separators; `—` for null/NaN. */
export function formatNumber(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/**
 * Parse a backend datetime. FastAPI emits naive datetimes (no offset) for
 * UTC columns; treat those as UTC rather than local time.
 */
export function parseApiDate(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const hasZone = /([zZ]|[+-]\d{2}:?\d{2})$/.test(iso);
  const d = new Date(hasZone ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

export async function copyToClipboard(text: string): Promise<void> {
  if (typeof navigator !== "undefined" && navigator.clipboard) {
    await navigator.clipboard.writeText(text);
  }
}
