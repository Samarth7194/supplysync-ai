import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { HistoricalReplayResponse, ModelMonitoringSnapshot, ModelMonitoringStatus } from "../lib/api.ts";
import {
  MODEL_MONITORING_ENDPOINT,
  MODEL_MONITORING_REPLAY_ENDPOINT,
  MODEL_MONITORING_STATUS_EXPLANATIONS,
  formatBaselineProvenance,
  formatForecastMethodLabel,
  formatMonitoringMetric,
  formatMonitoringStatus,
  formatReplayPeriod,
  formatSignedPercent,
  historicalReplayExplanation,
  monitoringExplanation,
  orderedMethodBreakdown,
  selectModelHealthEvidence,
  usesOfflineBacktestBaseline,
} from "../lib/modelMonitoring.ts";

function snapshot(overrides: Partial<ModelMonitoringSnapshot> = {}): ModelMonitoringSnapshot {
  return {
    model_name: "lightgbm_demand_forecast",
    model_version: "v1",
    lifecycle_status: "active",
    status: "stable",
    evaluation_count: 30,
    consecutive_degradation_count: 0,
    ...overrides,
  };
}

test("formats all supported monitoring states with truthful labels", () => {
  const cases: Array<[ModelMonitoringStatus, string]> = [
    ["unavailable", "Unavailable"],
    ["insufficient_evidence", "Insufficient Evidence"],
    ["stable", "Stable"],
    ["warning", "Warning"],
    ["degraded", "Degraded"],
  ];

  for (const [status, label] of cases) {
    assert.equal(formatMonitoringStatus(status), label);
    assert.ok(MODEL_MONITORING_STATUS_EXPLANATIONS[status].length > 0);
  }
});

test("uses backend degradation message when present", () => {
  assert.equal(
    monitoringExplanation(snapshot({ status: "warning", degradation_message: "Recent WAPE is 20% worse." })),
    "Recent WAPE is 20% worse.",
  );
});

test("falls back to state explanation for unavailable or missing snapshots", () => {
  assert.equal(monitoringExplanation(null), MODEL_MONITORING_STATUS_EXPLANATIONS.unavailable);
  assert.equal(
    monitoringExplanation(snapshot({ status: "insufficient_evidence" })),
    MODEL_MONITORING_STATUS_EXPLANATIONS.insufficient_evidence,
  );
});

test("humanizes baseline provenance and flags offline backtest baseline", () => {
  assert.equal(formatBaselineProvenance("promotion_evidence"), "Promotion Evidence");
  assert.equal(formatBaselineProvenance("artifact_metadata"), "Artifact Metadata");
  assert.equal(formatBaselineProvenance("offline_backtest"), "Offline Backtest");
  assert.equal(formatBaselineProvenance("unavailable"), "Unavailable");
  assert.equal(usesOfflineBacktestBaseline(snapshot({ baseline_provenance: "offline_backtest" })), true);
});

test("formats missing metrics safely", () => {
  assert.equal(formatMonitoringMetric(null), "—");
  assert.equal(formatSignedPercent(undefined), "—");
});

test("formats relative WAPE change and bias ratio as signed percentages", () => {
  assert.equal(formatSignedPercent(0.205), "+20.5%");
  assert.equal(formatSignedPercent(-0.082), "-8.2%");
  assert.equal(formatSignedPercent(0), "0.0%");
});

test("formats WAPE-like metrics consistently as decimals", () => {
  assert.equal(formatMonitoringMetric(0.944), "0.94");
  assert.equal(formatMonitoringMetric(1), "1.00");
});

test("monitoring API client uses the documented current snapshot endpoint", () => {
  assert.equal(MODEL_MONITORING_ENDPOINT, "/api/model-monitoring");
});

test("model health card includes loading and API-failure states", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.match(source, /Loading model monitoring/);
  assert.match(source, /Monitoring temporarily unavailable/);
});

test("model health card includes the offline-backtest provenance note", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.match(source, /Baseline is derived from offline evaluation, not live production history\./);
});

test("model health card exposes the primary monitoring metrics", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  for (const label of [
    "Recent WAPE",
    "Baseline WAPE",
    "Relative Change",
    "Bias Ratio",
    "Evaluations",
    "Last Run",
  ]) {
    assert.match(source, new RegExp(label));
  }
});

test("model health card uses responsive grid classes", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.match(source, /grid-cols-2 sm:grid-cols-3 lg:grid-cols-6/);
  assert.match(source, /grid-cols-1 md:grid-cols-2/);
});

test("monitoring UI does not advertise unimplemented MLOps actions or drift", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(source, /Retrain Model|Promote Model|Switch Model|Fix Drift/);
  assert.doesNotMatch(source, /Data Drift|Feature Drift|Concept Drift/);
});

// -- Historical replay precedence -------------------------------------------

function replay(overrides: Partial<HistoricalReplayResponse> = {}): HistoricalReplayResponse {
  return {
    mode: "historical_replay",
    available: true,
    status: "warning",
    metrics: { wape: 1.25 },
    baseline_wape: 1.07,
    evaluation_count: 39,
    sku_count: 59,
    horizon_days: 7,
    historical_period: { start: "2011-12-03", end: "2011-12-09" },
    provenance: "historical_replay",
    method_breakdown: {},
    live_monitoring: { available: false, evaluation_count: 0 },
    ...overrides,
  };
}

test("replay endpoint constant matches the documented read-only route", () => {
  assert.equal(MODEL_MONITORING_REPLAY_ENDPOINT, "/api/model-monitoring/replay");
});

test("live evidence takes precedence over historical replay when live has a classified state", () => {
  const live = snapshot({ status: "stable" });
  const evidence = selectModelHealthEvidence({ liveSnapshot: live, liveError: false, replay: replay() });
  assert.equal(evidence.mode, "live");
});

test("historical replay is shown when live monitoring is unavailable or insufficient", () => {
  for (const liveStatus of ["unavailable", "insufficient_evidence"] as ModelMonitoringStatus[]) {
    const live = liveStatus === "unavailable" ? null : snapshot({ status: liveStatus });
    const evidence = selectModelHealthEvidence({ liveSnapshot: live, liveError: false, replay: replay() });
    assert.equal(evidence.mode, "historical_replay");
  }
});

test("unavailable remains when neither live nor replay evidence exists", () => {
  const evidence = selectModelHealthEvidence({ liveSnapshot: null, liveError: false, replay: replay({ available: false }) });
  assert.equal(evidence.mode, "unavailable");
});

test("a live API error falls back to replay rather than showing stale live data", () => {
  const evidence = selectModelHealthEvidence({
    liveSnapshot: snapshot({ status: "stable" }),
    liveError: true,
    replay: replay(),
  });
  assert.equal(evidence.mode, "historical_replay");
});

test("live and replay evaluation counts are never combined", () => {
  const live = snapshot({ status: "insufficient_evidence", evaluation_count: 5 });
  const r = replay({ evaluation_count: 39 });
  const evidence = selectModelHealthEvidence({ liveSnapshot: live, liveError: false, replay: r });
  assert.equal(evidence.mode, "historical_replay");
  if (evidence.mode === "historical_replay") {
    assert.equal(evidence.replay.evaluation_count, 39);
    assert.notEqual(evidence.replay.evaluation_count, 5 + 39);
  }
});

test("historical replay explanation never claims live production truth", () => {
  const text = historicalReplayExplanation(replay());
  assert.doesNotMatch(text, /live production|real-time|ERP|POS/i);
});

test("formats the historical period as a readable range", () => {
  assert.equal(formatReplayPeriod({ start: "2011-12-03", end: "2011-12-09" }), "2011-12-03 → 2011-12-09");
  assert.equal(formatReplayPeriod(null), "—");
});

test("model health card visibly labels historical replay and never overclaims", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.match(source, /HISTORICAL REPLAY/);
  assert.match(source, /This is not live production monitoring\./);
  assert.match(source, /Live production actual-demand ingestion is not connected yet\./);
  assert.match(source, /Live Production Evidence/);
  assert.match(source, /completed evaluations/);
  assert.doesNotMatch(source, /live ERP evidence|real-time drift/i);
});

// -- Forecasting method breakdown --------------------------------------------

test("forecast method labels use customer-facing names, not internal keys", () => {
  assert.equal(formatForecastMethodLabel("ml_lightgbm"), "LightGBM");
  assert.equal(formatForecastMethodLabel("croston"), "Croston-SBA");
  assert.equal(formatForecastMethodLabel("conservative"), "Conservative");
});

test("unknown forecast methods still get a readable label instead of crashing", () => {
  assert.equal(formatForecastMethodLabel("simple_average"), "Simple Average");
  assert.equal(formatForecastMethodLabel("some_future_method"), "Some Future Method");
});

test("method breakdown never invents a method absent from the payload", () => {
  const entries = orderedMethodBreakdown({
    ml_lightgbm: { sku_count: 47, evaluation_count: 133, wape: 1.137 },
  });
  assert.equal(entries.length, 1);
  assert.equal(entries[0].method, "ml_lightgbm");
  assert.equal(entries[0].label, "LightGBM");

  assert.deepEqual(orderedMethodBreakdown({}), []);
  assert.deepEqual(orderedMethodBreakdown(null), []);
  assert.deepEqual(orderedMethodBreakdown(undefined), []);
});

test("method breakdown orders regular before intermittent before highly-intermittent", () => {
  const entries = orderedMethodBreakdown({
    conservative: { sku_count: 1, evaluation_count: 3, wape: 4.06 },
    ml_lightgbm: { sku_count: 47, evaluation_count: 133, wape: 1.14 },
    croston: { sku_count: 11, evaluation_count: 31, wape: 0.79 },
  });
  assert.deepEqual(
    entries.map((e) => e.method),
    ["ml_lightgbm", "croston", "conservative"],
  );
});

test("method breakdown surfaces only sku count, evaluation count, and WAPE — no invented metrics", () => {
  const entries = orderedMethodBreakdown({
    croston: { sku_count: 11, evaluation_count: 31, wape: 0.79 },
  });
  assert.deepEqual(Object.keys(entries[0]).sort(), ["evaluationCount", "label", "method", "skuCount", "wape"]);
});

test("model health card renders the forecasting method performance section without hardcoding methods", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.match(source, /Forecasting Method Performance/);
  assert.match(source, /orderedMethodBreakdown/);
  // Labels come from the shared lib mapping, not duplicated string literals
  // hardcoded into the component.
  assert.doesNotMatch(source, /"LightGBM"|"Croston-SBA"|"Conservative"/);
});

test("model health card explains the distinction between artifact health and method breakdown", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.match(source, /tracks the active LightGBM artifact/);
  assert.match(source, /replay performance/);
});

test("artifact-level Model Health metrics remain LightGBM-scoped even with a method breakdown present", () => {
  const source = readFileSync(new URL("../components/ModelHealthCard.tsx", import.meta.url), "utf8");
  assert.match(source, /LightGBM-artifact-scoped WAPE/);
  assert.match(source, /LightGBM-artifact-scoped evaluations/);
});
