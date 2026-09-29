import assert from "node:assert/strict";
import test from "node:test";

import {
  formatForecastMethod,
  formatDemandPattern,
  formatNumber,
  formatUnits,
  computeDemoStock,
} from "../lib/utils.ts";

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

test("computeDemoStock is deterministic per SKU id, not per list position", () => {
  const a1 = computeDemoStock(10, 2, "SKU-A");
  const a2 = computeDemoStock(10, 2, "SKU-A");
  assert.equal(a1, a2);
});

test("computeDemoStock spreads SKUs across below/near/above their own reorder point", () => {
  // A pool of distinct ids should land in more than one bucket relative to
  // their own (avgDemand, demandStd)-derived reorder point -- the whole
  // point of replacing the old index%3 generator was that some SKUs must
  // show NO_ACTION (stock comfortably above reorder point) and some must
  // show a reorder is needed (stock below it).
  const avgDemand = 20;
  const demandStd = 4;
  const leadTimeDemand = avgDemand * 7;
  const safetyStock = 1.645 * demandStd * Math.sqrt(7);
  const reorderPoint = leadTimeDemand + safetyStock;

  const ids = Array.from({ length: 30 }, (_, i) => `SKU-${i}`);
  const stocks = ids.map((id) => computeDemoStock(avgDemand, demandStd, id));

  const below = stocks.filter((s) => s < reorderPoint * 0.9);
  const above = stocks.filter((s) => s > reorderPoint * 1.1);

  assert.ok(below.length > 0, "expected at least one SKU below its reorder point");
  assert.ok(above.length > 0, "expected at least one SKU comfortably above its reorder point (NO_ACTION)");
});

test("computeDemoStock never returns a negative stock level", () => {
  for (let i = 0; i < 20; i++) {
    const stock = computeDemoStock(0, 0, `EDGE-${i}`);
    assert.ok(stock >= 0);
  }
});
