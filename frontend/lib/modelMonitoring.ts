import type { HistoricalReplayResponse, ModelMonitoringSnapshot, ModelMonitoringStatus } from "./api";

export const MONITORING_DASH = "—";
export const MODEL_MONITORING_ENDPOINT = "/api/model-monitoring";
export const MODEL_MONITORING_REPLAY_ENDPOINT = "/api/model-monitoring/replay";

export const MODEL_MONITORING_STATUS_LABELS: Record<ModelMonitoringStatus, string> = {
  unavailable: "Unavailable",
  insufficient_evidence: "Insufficient Evidence",
  stable: "Stable",
  warning: "Warning",
  degraded: "Degraded",
};

export const MODEL_MONITORING_STATUS_EXPLANATIONS: Record<ModelMonitoringStatus, string> = {
  unavailable: "Monitoring has not produced a snapshot for the current model yet.",
  insufficient_evidence: "More completed forecast evaluations are required before performance can be classified.",
  stable: "Recent forecast performance is within the configured range of the baseline.",
  warning: "Recent forecast performance is worse than baseline and requires attention.",
  degraded: "Forecast degradation persisted across multiple monitoring runs.",
};

export const MODEL_MONITORING_STATUS_STYLES: Record<
  ModelMonitoringStatus,
  { badge: string; dot: string; border: string }
> = {
  unavailable: {
    badge: "bg-gray-500/10 text-gray-400 border-gray-500/30",
    dot: "bg-gray-500",
    border: "border-gray-800",
  },
  insufficient_evidence: {
    badge: "bg-blue-500/10 text-blue-300 border-blue-500/30",
    dot: "bg-blue-300",
    border: "border-blue-500/20",
  },
  stable: {
    badge: "bg-green-500/10 text-green-300 border-green-500/30",
    dot: "bg-green-300",
    border: "border-green-500/20",
  },
  warning: {
    badge: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    dot: "bg-amber-300",
    border: "border-amber-500/30",
  },
  degraded: {
    badge: "bg-red-500/10 text-red-300 border-red-500/30",
    dot: "bg-red-300",
    border: "border-red-500/30",
  },
};

const BASELINE_LABELS: Record<string, string> = {
  promotion_evidence: "Promotion Evidence",
  artifact_metadata: "Artifact Metadata",
  offline_backtest: "Offline Backtest",
  unavailable: "Unavailable",
};

export function formatMonitoringStatus(status: ModelMonitoringStatus | null | undefined): string {
  return status ? MODEL_MONITORING_STATUS_LABELS[status] : MODEL_MONITORING_STATUS_LABELS.unavailable;
}

export function monitoringExplanation(snapshot: ModelMonitoringSnapshot | null | undefined): string {
  if (!snapshot) return MODEL_MONITORING_STATUS_EXPLANATIONS.unavailable;
  return snapshot.degradation_message || MODEL_MONITORING_STATUS_EXPLANATIONS[snapshot.status];
}

export function formatBaselineProvenance(value: string | null | undefined): string {
  if (!value) return MONITORING_DASH;
  return BASELINE_LABELS[value] ?? value.replace(/_/g, " ");
}

export function usesOfflineBacktestBaseline(snapshot: ModelMonitoringSnapshot | null | undefined): boolean {
  return snapshot?.baseline_provenance === "offline_backtest";
}

export function formatMonitoringMetric(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return MONITORING_DASH;
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatSignedPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return MONITORING_DASH;
  const sign = value > 0 ? "+" : "";
  return `${sign}${new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value * 100)}%`;
}

export function formatMonitoringTime(value: string | null | undefined): string {
  if (!value) return MONITORING_DASH;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// -- Historical replay ------------------------------------------------------
//
// Historical replay is a held-out historical-window backtest of the same
// forecast -> evaluate -> monitor lifecycle, generated offline because the
// processed dataset is historical and there is no connected ERP/POS
// actual-demand stream. It must never be confused with live monitoring —
// see `selectModelHealthEvidence` for the display precedence rule.

export type ModelHealthEvidence =
  | { mode: "live"; snapshot: ModelMonitoringSnapshot }
  | { mode: "historical_replay"; replay: HistoricalReplayResponse }
  | { mode: "unavailable" };

/**
 * Decide what the Model Health card should display.
 *
 * Precedence (never combined):
 *   1. LIVE monitoring, only when a real snapshot exists AND it has reached
 *      a classified state (stable/warning/degraded) — insufficient_evidence
 *      is honest but not "sufficient valid evidence", so it falls through.
 *   2. HISTORICAL REPLAY, only when live evidence is not sufficient and a
 *      replay result is available.
 *   3. Unavailable.
 */
export function selectModelHealthEvidence(params: {
  liveSnapshot: ModelMonitoringSnapshot | null | undefined;
  liveError: boolean;
  replay: HistoricalReplayResponse | null | undefined;
}): ModelHealthEvidence {
  const { liveSnapshot, liveError, replay } = params;
  const liveHasSufficientEvidence =
    !liveError &&
    !!liveSnapshot &&
    (["stable", "warning", "degraded"] as ModelMonitoringStatus[]).includes(liveSnapshot.status);

  if (liveHasSufficientEvidence && liveSnapshot) {
    return { mode: "live", snapshot: liveSnapshot };
  }
  if (replay?.available) {
    return { mode: "historical_replay", replay };
  }
  return { mode: "unavailable" };
}

export function historicalReplayExplanation(replay: HistoricalReplayResponse | null | undefined): string {
  if (!replay?.available) return "No historical replay has been generated yet.";
  return (
    replay.degradation_message ||
    "Historical replay demonstrates model monitoring using held-out historical demand."
  );
}

export function formatReplayPeriod(period: { start?: string | null; end?: string | null } | null | undefined): string {
  if (!period?.start || !period?.end) return MONITORING_DASH;
  return `${period.start} → ${period.end}`;
}

// -- Forecasting method breakdown --------------------------------------------
//
// SupplySync routes each SKU to one of several forecasting methods based on
// its demand pattern (regular -> LightGBM, intermittent -> Croston-SBA,
// highly intermittent -> a conservative buffer). Model Health itself stays
// scoped to the active LightGBM artifact; this mapping only controls how the
// *separate*, informational method-breakdown section labels whatever methods
// the replay payload actually reports — it never fabricates a method that
// isn't present in the data.

const FORECAST_METHOD_LABELS: Record<string, string> = {
  ml_lightgbm: "LightGBM",
  croston: "Croston-SBA",
  conservative: "Conservative",
  simple_average: "Simple Average",
};

// Preferred reading order (regular -> intermittent -> highly intermittent).
// Only methods actually present in the payload are ever shown; this just
// controls ordering when they are.
const FORECAST_METHOD_ORDER = ["ml_lightgbm", "croston", "conservative", "simple_average"];

export function formatForecastMethodLabel(method: string): string {
  return (
    FORECAST_METHOD_LABELS[method] ??
    method
      .split("_")
      .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
      .join(" ")
  );
}

export interface MethodBreakdownEntry {
  method: string;
  label: string;
  skuCount: number;
  evaluationCount: number;
  wape: number | null;
}

/**
 * Turn the raw `method_breakdown` payload into an ordered, labeled list for
 * display. Only methods actually present in `breakdown` are returned — this
 * never invents a method the API didn't report (e.g. no Croston card renders
 * if no SKU was ever routed to Croston in the replayed windows).
 */
export function orderedMethodBreakdown(
  breakdown: Record<string, { sku_count: number; evaluation_count: number; wape: number | null }> | null | undefined,
): MethodBreakdownEntry[] {
  if (!breakdown) return [];
  const keys = Object.keys(breakdown);
  const ordered = [
    ...FORECAST_METHOD_ORDER.filter((key) => keys.includes(key)),
    ...keys.filter((key) => !FORECAST_METHOD_ORDER.includes(key)).sort(),
  ];
  return ordered.map((method) => ({
    method,
    label: formatForecastMethodLabel(method),
    skuCount: breakdown[method].sku_count,
    evaluationCount: breakdown[method].evaluation_count,
    wape: breakdown[method].wape,
  }));
}
