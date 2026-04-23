import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Short relative-time string ("just now", "12m ago", "3h ago", "2d ago").
 * Falls back to ISO date for anything older than ~30 days so we don't show
 * misleading "350d ago"-style labels in activity tables.
 */
export function formatRelativeTime(iso: string | Date): string {
  const then = iso instanceof Date ? iso : new Date(iso);
  if (Number.isNaN(then.getTime())) return String(iso);

  const deltaMs = Date.now() - then.getTime();
  const deltaSec = Math.max(0, Math.round(deltaMs / 1000));
  if (deltaSec < 45) return "just now";
  const deltaMin = Math.round(deltaSec / 60);
  if (deltaMin < 60) return `${deltaMin}m ago`;
  const deltaHr = Math.round(deltaMin / 60);
  if (deltaHr < 24) return `${deltaHr}h ago`;
  const deltaDay = Math.round(deltaHr / 24);
  if (deltaDay < 30) return `${deltaDay}d ago`;
  return then.toISOString().slice(0, 10);
}
