import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { ModelMonitoringSnapshot, ModelMonitoringStatus } from "../lib/api.ts";
import {
  MODEL_MONITORING_ENDPOINT,
  MODEL_MONITORING_STATUS_EXPLANATIONS,
  formatBaselineProvenance,
  formatMonitoringMetric,
  formatMonitoringStatus,
  formatSignedPercent,
  monitoringExplanation,
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
