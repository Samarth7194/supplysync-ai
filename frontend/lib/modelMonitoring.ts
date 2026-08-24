import type { ModelMonitoringSnapshot, ModelMonitoringStatus } from "./api";

export const MONITORING_DASH = "—";
export const MODEL_MONITORING_ENDPOINT = "/api/model-monitoring";

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
