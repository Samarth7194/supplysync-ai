import assert from "node:assert/strict";
import test from "node:test";

import { formatForecastMethod, formatDemandPattern, formatNumber, formatUnits } from "../lib/utils.ts";

test("forecast method labels match the customer-facing names used elsewhere in the UI", () => {
  // Regression guard: this label used to say plain "Croston" here while the
  // Model Health method-breakdown section (lib/modelMonitoring.ts) said
  // "Croston-SBA" for the exact same internal method key — the kind of
  // inconsistency a reviewer notices immediately. Both surfaces must agree.
  assert.equal(formatForecastMethod("ml_lightgbm"), "LightGBM");
  assert.equal(formatForecastMethod("croston"), "Croston-SBA");
  assert.equal(formatForecastMethod("conservative"), "Conservative");
  assert.equal(formatForecastMethod("simple_average"), "Simple Average");
});

test("forecast method label falls back to the raw method string instead of crashing", () => {
  assert.equal(formatForecastMethod("some_future_method"), "some_future_method");
  assert.equal(formatForecastMethod(null), "");
  assert.equal(formatForecastMethod(undefined), "");
});

test("demand pattern labels are human-readable", () => {
  assert.equal(formatDemandPattern("regular"), "Regular");
  assert.equal(formatDemandPattern("intermittent"), "Intermittent");
  assert.equal(formatDemandPattern("highly_intermittent"), "Highly Intermittent");
});

test("formatNumber and formatUnits render a dash instead of NaN/undefined text", () => {
  assert.equal(formatNumber(null), "—");
  assert.equal(formatNumber(undefined), "—");
  assert.equal(formatNumber(Number.NaN), "—");
  assert.equal(formatUnits(null), "—");
  assert.equal(formatUnits(42), "42 units");
});
