// Per-SKU current-stock override, persisted in browser localStorage.
//
// The dashboard falls back to a deterministic demo value (avg_demand * multiplier)
// when no override is set. Users can edit the value from the SKU detail page; the
// override persists across reloads in the same browser only. There is no server-side
// inventory store — this is intentionally a local-demo pathway, and the UI labels it
// ("Stored locally" vs. "Demo value") so it's never mistaken for live inventory.

const STOCK_KEY = (sku: string) => `supplysync.stock.v1.${sku}`;
const ORIGIN_KEY = (sku: string) => `supplysync.stock-origin.v1.${sku}`;

export type StockOrigin = "user" | "demo";

export function getStockForSku(sku: string, fallback: number): number {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(STOCK_KEY(sku));
    if (raw === null) return fallback;
    const parsed = Number(raw);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
  } catch {
    return fallback;
  }
}

export function setStockForSku(sku: string, value: number): void {
  if (typeof window === "undefined") return;
  const safe = Math.max(0, Math.round(value));
  try {
    window.localStorage.setItem(STOCK_KEY(sku), String(safe));
    window.localStorage.setItem(ORIGIN_KEY(sku), "user");
  } catch {
    // Quota or disabled storage — silently ignore; fallback path still works.
  }
}

export function getStockOrigin(sku: string): StockOrigin {
  if (typeof window === "undefined") return "demo";
  try {
    return window.localStorage.getItem(ORIGIN_KEY(sku)) === "user" ? "user" : "demo";
  } catch {
    return "demo";
  }
}

export function clearStockForSku(sku: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(STOCK_KEY(sku));
    window.localStorage.removeItem(ORIGIN_KEY(sku));
  } catch {
    // ignore
  }
}
