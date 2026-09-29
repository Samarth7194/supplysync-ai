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

export function formatNumber(
  value: number | null | undefined,
  options: Intl.NumberFormatOptions = {},
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", options).format(value);
}

export function formatUnits(
  value: number | null | undefined,
  options: Intl.NumberFormatOptions = {},
): string {
  const formatted = formatNumber(value, options);
  return formatted === "—" ? formatted : `${formatted} units`;
}

export function formatForecastMethod(method: string | null | undefined): string {
  const labels: Record<string, string> = {
    ml_lightgbm: "LightGBM",
    croston: "Croston-SBA",
    conservative: "Conservative",
    simple_average: "Simple Average",
  };
  return method ? labels[method] ?? method : "";
}

export function formatDemandPattern(pattern: string | null | undefined): string {
  const labels: Record<string, string> = {
    regular: "Regular",
    intermittent: "Intermittent",
    highly_intermittent: "Highly Intermittent",
  };
  return pattern ? labels[pattern] ?? pattern : "";
}

// Deterministic demo stock levels.
//
// Seeded from an approximation of the SKU's own inventory position (the same
// lead-time-demand + safety-stock reorder-point formula the backend uses),
// not a plain multiple of average demand -- so, deterministically per SKU
// (hashed on its id, not its position in the list), roughly a third of
// demo SKUs land above their own reorder point and show NO_ACTION, a third
// land right at it, and a third land below it and need a reorder.
const DEMO_SERVICE_LEVEL_Z = 1.645; // 95% one-sided service level, matches the backend
const DEMO_LEAD_TIME_DAYS = 7;

function _stableHash(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

export function computeDemoStock(avgDemand: number, demandStd: number, skuId: string): number {
  const leadTimeDemand = avgDemand * DEMO_LEAD_TIME_DAYS;
  const safetyStock = DEMO_SERVICE_LEVEL_Z * demandStd * Math.sqrt(DEMO_LEAD_TIME_DAYS);
  const reorderPoint = leadTimeDemand + safetyStock;

  const bucket = _stableHash(skuId) % 3;
  const multipliers = [0.4, 1.0, 1.8]; // below, near, comfortably above the reorder point
  return Math.max(0, Math.round(reorderPoint * multipliers[bucket]));
}
