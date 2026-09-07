"use client";

import React from "react";
import { Activity, AlertCircle, History } from "lucide-react";
import type { HistoricalReplayResponse, ModelMonitoringSnapshot } from "@/lib/api";
import {
  MODEL_MONITORING_STATUS_STYLES,
  formatBaselineProvenance,
  formatMonitoringMetric,
  formatMonitoringStatus,
  formatMonitoringTime,
  formatReplayPeriod,
  formatSignedPercent,
  historicalReplayExplanation,
  monitoringExplanation,
  orderedMethodBreakdown,
  selectModelHealthEvidence,
  usesOfflineBacktestBaseline,
} from "@/lib/modelMonitoring";
import { formatNumber } from "@/lib/utils";

function Metric({
  label,
  value,
  title,
}: {
  label: string;
  value: React.ReactNode;
  title?: string;
}) {
  return (
    <div className="min-w-0" title={title}>
      <p className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">{label}</p>
      <p className="mt-1 text-sm font-semibold text-white tabular-nums truncate">{value}</p>
    </div>
  );
}

function DetailRow({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-xs" title={hint}>
      <span className="text-gray-500 uppercase tracking-wider">{label}</span>
      <span className="text-gray-300 text-right break-words">{value}</span>
    </div>
  );
}

/**
 * Shared card header for both Model Health variants (live + historical
 * replay). Keeping this in one place means the eyebrow/title/subtitle
 * treatment can never silently drift between the two states.
 */
function ModelHealthHeader({
  subtitle,
  badges,
}: {
  subtitle: React.ReactNode;
  badges: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 mb-4">
      <div>
        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.12em] mb-1.5">
          Performance Monitoring
        </p>
        <h2 className="font-semibold text-white text-base">Model Health</h2>
        <p className="text-xs text-gray-500 mt-1 max-w-2xl leading-relaxed">{subtitle}</p>
      </div>
      <div className="flex flex-col items-end gap-1.5 shrink-0">{badges}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: keyof typeof MODEL_MONITORING_STATUS_STYLES }) {
  const styles = MODEL_MONITORING_STATUS_STYLES[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-semibold ${styles.badge}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${styles.dot}`} aria-hidden="true" />
      {formatMonitoringStatus(status)}
    </span>
  );
}

function MethodPerformanceCard({ entry }: { entry: ReturnType<typeof orderedMethodBreakdown>[number] }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-950/60 p-3 min-w-0">
      <p className="text-xs font-semibold text-white truncate">{entry.label}</p>
      <p className="mt-1.5 text-[11px] text-gray-500">
        {formatNumber(entry.skuCount, { maximumFractionDigits: 0 })} SKU{entry.skuCount === 1 ? "" : "s"}
      </p>
      <p className="mt-2 text-[10px] font-medium text-gray-500 uppercase tracking-wider">WAPE</p>
      <p className="text-sm font-semibold text-white tabular-nums">{formatMonitoringMetric(entry.wape)}</p>
    </div>
  );
}

function MethodBreakdownSection({ replay }: { replay: HistoricalReplayResponse }) {
  const entries = orderedMethodBreakdown(replay.method_breakdown);
  if (entries.length === 0) return null;

  return (
    <div className="border-t border-gray-800 pt-3 space-y-2">
      <div>
        <p className="text-xs font-semibold text-white">Forecasting Method Performance</p>
        <p className="text-[11px] text-gray-500 mt-1 leading-relaxed max-w-2xl">
          SupplySync routes SKUs to different forecasting methods based on demand behavior. Model
          Health above tracks the active LightGBM artifact; this breakdown shows replay performance
          across all forecasting methods, across every replayed window.
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {entries.map((entry) => (
          <MethodPerformanceCard key={entry.method} entry={entry} />
        ))}
      </div>
    </div>
  );
}

function HistoricalReplayCard({ replay }: { replay: HistoricalReplayResponse }) {
  const status = replay.status ?? "insufficient_evidence";
  const styles = MODEL_MONITORING_STATUS_STYLES[status];
  const explanation = historicalReplayExplanation(replay);

  return (
    <section className={`bg-gray-900 border ${styles.border} rounded-xl p-5`}>
      <ModelHealthHeader
        subtitle="Based on historical holdout replay. This is not live production monitoring."
        badges={
          <>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-semibold bg-purple-500/10 text-purple-300 border-purple-500/30">
              <History className="w-3 h-3" aria-hidden="true" />
              HISTORICAL REPLAY
            </span>
            <StatusBadge status={status} />
          </>
        }
      />

      <div className="space-y-4">
        <div className="flex items-start gap-2 text-xs text-gray-300 leading-relaxed">
          <History className="w-4 h-4 mt-0.5 text-purple-300 shrink-0" aria-hidden="true" />
          <p>{explanation}</p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <Metric
            label="Replay WAPE"
            value={formatMonitoringMetric(replay.metrics.wape)}
            title="Weighted Absolute Percentage Error (sum of absolute errors / sum of actual demand). This is LightGBM-artifact-scoped WAPE for the most recent replay window. Lower is better."
          />
          <Metric
            label="Baseline WAPE"
            value={formatMonitoringMetric(replay.baseline_wape)}
            title="The offline-backtest WAPE this replay window is compared against, matched by forecast horizon."
          />
          <Metric
            label="Evaluations"
            value={formatNumber(replay.evaluation_count, { maximumFractionDigits: 0 })}
            title="LightGBM-artifact-scoped evaluations in the most recent replay window only — this is the count the status above is based on."
          />
          <Metric
            label="SKUs"
            value={formatNumber(replay.sku_count, { maximumFractionDigits: 0 })}
            title="Unique SKUs covered across every replayed window and every routing method (LightGBM, Croston-SBA, conservative) — broader than, and not directly comparable to, the Evaluations count."
          />
          <Metric label="Horizon (days)" value={formatNumber(replay.horizon_days, { maximumFractionDigits: 0 })} />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 border-t border-gray-800 pt-3">
          <DetailRow label="Model" value={replay.model_name || "—"} />
          <DetailRow label="Version" value={replay.model_version || "—"} />
          <DetailRow label="Historical Period" value={formatReplayPeriod(replay.historical_period)} />
          <DetailRow
            label="Baseline"
            value={formatBaselineProvenance(replay.baseline_provenance)}
            hint="Where the comparison WAPE came from: promotion evidence, artifact training metrics, or an offline backtest."
          />
          <DetailRow label="Reason" value={replay.degradation_reason?.replace(/_/g, " ") || "-"} />
          <DetailRow
            label="Live Production Evidence"
            value={`${formatNumber(replay.live_monitoring?.evaluation_count ?? 0, {
              maximumFractionDigits: 0,
            })} completed evaluations / ${replay.live_monitoring?.available ? "available" : "unavailable"}`}
          />
          {typeof replay.metrics.mae === "number" && (
            <DetailRow label="MAE" value={formatMonitoringMetric(replay.metrics.mae)} hint="Mean Absolute Error, in demand units." />
          )}
          {typeof replay.metrics.rmse === "number" && (
            <DetailRow label="RMSE" value={formatMonitoringMetric(replay.metrics.rmse)} hint="Root Mean Squared Error — penalizes large misses more than MAE." />
          )}
          {typeof replay.metrics.mase === "number" && (
            <DetailRow
              label="MASE"
              value={formatMonitoringMetric(replay.metrics.mase)}
              hint="Mean Absolute Scaled Error — below 1.0 means this beats a naive repeat-last-value forecast."
            />
          )}
          <DetailRow label="Generated" value={formatMonitoringTime(replay.generated_at)} />
        </div>

        <MethodBreakdownSection replay={replay} />

        <p className="rounded-lg border border-purple-500/20 bg-purple-500/10 px-3 py-2 text-xs text-purple-200 leading-relaxed">
          Historical replay demonstrates model monitoring using held-out historical demand.
          Live production actual-demand ingestion is not connected yet.
        </p>
      </div>
    </section>
  );
}

export function ModelHealthCard({
  snapshot,
  loading,
  error,
  replay = null,
}: {
  snapshot: ModelMonitoringSnapshot | null;
  loading: boolean;
  error: boolean;
  replay?: HistoricalReplayResponse | null;
}) {
  const evidence = selectModelHealthEvidence({ liveSnapshot: snapshot, liveError: error, replay });

  if (!loading && evidence.mode === "historical_replay") {
    return <HistoricalReplayCard replay={evidence.replay} />;
  }

  const status = snapshot?.status ?? "unavailable";
  const styles = MODEL_MONITORING_STATUS_STYLES[status];
  const explanation = error
    ? "Monitoring temporarily unavailable"
    : monitoringExplanation(snapshot);

  return (
    <section className={`bg-gray-900 border ${styles.border} rounded-xl p-5`}>
      <ModelHealthHeader
        subtitle="Latest completed forecast-evaluation snapshot for the active demand model."
        badges={
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-semibold ${styles.badge}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${styles.dot}`} aria-hidden="true" />
            {error ? "Unavailable" : formatMonitoringStatus(status)}
          </span>
        }
      />

      {loading ? (
        <div aria-label="Loading model monitoring" className="space-y-3">
          <div className="h-4 w-2/3 bg-gray-800 rounded animate-pulse" />
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Array.from({ length: 5 }).map((_, index) => (
              <div key={index} className="space-y-2">
                <div className="h-3 w-16 bg-gray-800 rounded animate-pulse" />
                <div className="h-4 w-20 bg-gray-800 rounded animate-pulse" />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-start gap-2 text-xs text-gray-300 leading-relaxed">
            {error ? (
              <AlertCircle className="w-4 h-4 mt-0.5 text-amber-300 shrink-0" aria-hidden="true" />
            ) : (
              <Activity className="w-4 h-4 mt-0.5 text-blue-300 shrink-0" aria-hidden="true" />
            )}
            <p>{explanation}</p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            <Metric
              label="Recent WAPE"
              value={formatMonitoringMetric(snapshot?.metric_wape)}
              title="Weighted Absolute Percentage Error over the current rolling evaluation window. Lower is better."
            />
            <Metric label="Baseline WAPE" value={formatMonitoringMetric(snapshot?.baseline_wape)} />
            <Metric
              label="Relative Change"
              value={formatSignedPercent(snapshot?.wape_relative_change)}
              title="How much recent WAPE differs from the baseline — positive means performance got worse."
            />
            <Metric
              label="Bias Ratio"
              value={formatSignedPercent(snapshot?.bias_ratio)}
              title="Mean forecast error as a share of mean actual demand — a supporting warning signal, not a degradation trigger on its own."
            />
            <Metric label="Evaluations" value={formatNumber(snapshot?.evaluation_count, { maximumFractionDigits: 0 })} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 border-t border-gray-800 pt-3">
            <DetailRow label="Current model" value={snapshot?.model_name || "—"} />
            <DetailRow label="Lifecycle" value={snapshot?.lifecycle_status || "—"} />
            <DetailRow label="Last Run" value={formatMonitoringTime(snapshot?.generated_at)} />
            <DetailRow
              label="Baseline"
              value={formatBaselineProvenance(snapshot?.baseline_provenance)}
              hint="Where the comparison WAPE came from: promotion evidence, artifact training metrics, or an offline backtest."
            />
            <DetailRow label="Reason" value={snapshot?.degradation_reason?.replace(/_/g, " ") || "—"} />
            {typeof snapshot?.metric_mae === "number" && (
              <DetailRow label="MAE" value={formatMonitoringMetric(snapshot.metric_mae)} hint="Mean Absolute Error, in demand units." />
            )}
            {typeof snapshot?.metric_rmse === "number" && (
              <DetailRow label="RMSE" value={formatMonitoringMetric(snapshot.metric_rmse)} hint="Root Mean Squared Error — penalizes large misses more than MAE." />
            )}
            {typeof snapshot?.metric_mase === "number" && (
              <DetailRow
                label="MASE"
                value={formatMonitoringMetric(snapshot.metric_mase)}
                hint="Mean Absolute Scaled Error — below 1.0 means this beats a naive repeat-last-value forecast."
              />
            )}
            {typeof snapshot?.residual_std === "number" && (
              <DetailRow label="Residual Std" value={formatMonitoringMetric(snapshot.residual_std)} />
            )}
          </div>

          {usesOfflineBacktestBaseline(snapshot) && (
            <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-200 leading-relaxed">
              Baseline is derived from offline evaluation, not live production history.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
