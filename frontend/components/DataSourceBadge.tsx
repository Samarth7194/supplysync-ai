"use client";

import React from "react";
import type { DemandSource, ForecastSource } from "@/lib/api";

// Provenance vocabulary — keep in sync with backend main.py._classify_forecast_source
// and the demand_source values returned by /api/analyze.
export type ProvenanceKind =
  | "recorded"
  | "user_provided"
  | "user_stored"
  | "server_stored"
  | "synthetic"
  | "model_forecast"
  | "statistical_method"
  | "rule_based_estimate"
  | "demo_value"
  | "unavailable";

const COPY: Record<ProvenanceKind, { label: string; tooltip: string; className: string }> = {
  recorded: {
    label: "Recorded",
    tooltip: "Real historical demand from the processed dataset.",
    className: "bg-blue-500/10 text-blue-300 border-blue-500/30",
  },
  user_provided: {
    label: "User input",
    tooltip: "Demand history was supplied in the analyze request.",
    className: "bg-sky-500/10 text-sky-300 border-sky-500/30",
  },
  user_stored: {
    label: "Stored locally",
    tooltip: "You've set this value. It's stored in your browser's localStorage, not on any server.",
    className: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
  },
  server_stored: {
    label: "Server stored",
    tooltip: "This value is stored by the backend stock API.",
    className: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
  },
  synthetic: {
    label: "Synthetic demo",
    tooltip: "This SKU has no recorded history; values are synthetic demo data.",
    className: "bg-amber-500/10 text-amber-300 border-amber-500/30",
  },
  model_forecast: {
    label: "Model forecast",
    tooltip: "Produced by the trained LightGBM model.",
    className: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
  },
  statistical_method: {
    label: "Statistical",
    tooltip: "Produced by a statistical method (Croston or conservative buffer).",
    className: "bg-violet-500/10 text-violet-300 border-violet-500/30",
  },
  rule_based_estimate: {
    label: "Rule-based",
    tooltip: "Fallback estimate (simple moving average) — not a model forecast.",
    className: "bg-amber-500/10 text-amber-300 border-amber-500/30",
  },
  demo_value: {
    label: "Demo value",
    tooltip: "Illustrative value derived for the demo — not a real operational reading.",
    className: "bg-amber-500/10 text-amber-300 border-amber-500/30",
  },
  unavailable: {
    label: "Unavailable",
    tooltip: "Data is not available from the backend.",
    className: "bg-gray-500/10 text-gray-400 border-gray-500/30",
  },
};

export function DataSourceBadge({
  kind,
  className = "",
}: {
  kind: ProvenanceKind;
  className?: string;
}) {
  const entry = COPY[kind];
  return (
    <span
      title={entry.tooltip}
      className={`inline-flex items-center text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${entry.className} ${className}`}
    >
      {entry.label}
    </span>
  );
}

export function demandSourceKind(source: DemandSource | undefined): ProvenanceKind {
  switch (source) {
    case "historical":
      return "recorded";
    case "request":
      return "user_provided";
    case "synthetic":
      return "synthetic";
    default:
      return "unavailable";
  }
}

export function forecastSourceKind(
  source: ForecastSource | undefined,
): ProvenanceKind {
  switch (source) {
    case "model_forecast":
      return "model_forecast";
    case "statistical_method":
      return "statistical_method";
    case "rule_based_estimate":
      return "rule_based_estimate";
    default:
      return "unavailable";
  }
}
