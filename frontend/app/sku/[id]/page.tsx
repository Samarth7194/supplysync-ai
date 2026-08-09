"use client";

import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { ArrowLeft, Tags, TrendingUp, Shield, CheckCircle, AlertTriangle } from "lucide-react";
import {
  DataSourceBadge,
  demandSourceKind,
  forecastSourceKind,
} from "@/components/DataSourceBadge";
import type {
  DemandSource,
  ForecastSource,
  DecisionBlock,
  ModelInfo,
  ExplanationBlock,
} from "@/lib/api";
import { EmptyState } from "@/components/EmptyState";
import { InfoRow } from "@/components/InfoRow";
import { SectionHeader } from "@/components/SectionHeader";
import { env } from "@/lib/env";
import {
  getStockForSku,
  setStockForSku,
  getStockOrigin,
  clearStockForSku,
  type StockOrigin,
} from "@/lib/stock";
import { formatDemandPattern, formatForecastMethod } from "@/lib/utils";

const API = env.apiUrl;
const HISTORY_DAYS = 30;

function computeDemoStock(avgDemand: number, index: number): number {
  const multipliers = [0.3, 0.8, 2.0];
  return Math.max(1, Math.round(avgDemand * multipliers[index % 3]));
}

function formatShortDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 10); // YYYY-MM-DD, tz-stable
}

interface SkuDetail {
  id: string;
  name: string;
  avg_demand: number;
  total_demand: number;
}

interface Analysis {
  sku: string;
  risk: string;
  risk_color: string;
  forecast: {
    p50: number;
    p90: number;
    daily?: number[];
    full_horizon_daily?: number[];
    horizon_days?: number;
  };
  current_stock: number;
  recommended_order: number;
  action: string;
  demand_pattern: string;
  forecast_method: string;
  demand_source?: DemandSource;
  forecast_source?: ForecastSource;
  decision?: DecisionBlock;
  model_info?: ModelInfo;
  explanation?: ExplanationBlock;
}

function formatUnits(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${Math.round(value).toLocaleString()} units`;
}

interface HistoryPoint {
  date: string;
  demand: number;
  units_sold?: number;
}

interface HistorySummary {
  first_date: string;
  last_date: string;
  window_days_returned: number;
  total_units_sold: number;
  mean_units_per_day: number;
  peak_units_in_one_day: number;
  days_with_sales: number;
  days_with_zero_sales: number;
}

interface HistoryResponse {
  sku: string;
  available: boolean;
  history: HistoryPoint[];
  series_type?: string;
  value_meaning?: string;
  source?: string;
  description?: string;
  summary?: HistorySummary | null;
}

interface StockResponse {
  sku: string;
  quantity_on_hand: number;
  quantity_reserved: number;
  quantity_available: number;
  source: string;
  recorded_at?: string | null;
}

function RiskBadge({ risk }: { risk: string }) {
  const styles: Record<string, string> = {
    HIGH: "bg-red-500/15 text-red-400 border-red-500/30",
    MEDIUM: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30",
    LOW: "bg-green-500/15 text-green-400 border-green-500/30",
  };
  return (
    <span className={`text-sm font-semibold px-3 py-1 rounded-full border ${styles[risk] || styles.LOW}`}>
      {risk} RISK
    </span>
  );
}

const PATTERN_REASONS: Record<string, string> = {
  regular: "Less than 50% zero-demand days, consistent purchase pattern",
  intermittent: "50-80% zero-demand days, sporadic but recurring purchases",
  highly_intermittent: "Over 80% zero-demand days, very rare purchases",
};

export default function SKUDetail() {
  const params = useParams();
  const skuId = params.id as string;

  const [skuInfo, setSkuInfo] = useState<SkuDetail | null>(null);
  const [skuIndex, setSkuIndex] = useState(0);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [historyAvailable, setHistoryAvailable] = useState<boolean>(false);
  const [historySummary, setHistorySummary] = useState<HistorySummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Configurable business assumptions. Defaults match the backend defaults.
  const [leadTimeDays, setLeadTimeDays] = useState<number>(7);
  const [serviceLevelPct, setServiceLevelPct] = useState<number>(95);
  const [reAnalyzing, setReAnalyzing] = useState(false);

  // Current stock comes from the stock API when available, with a browser
  // fallback for local demo use.
  const [currentStock, setCurrentStock] = useState<number>(0);
  const [stockOrigin, setStockOrigin] = useState<StockOrigin>("demo");
  const [stockDraft, setStockDraft] = useState<string>("");

  async function fetchAnalysis(stock: number, lead: number, svcLevelPct: number) {
    const body = {
      sku: skuId,
      current_stock: stock,
      lead_time_days: lead,
      service_level: svcLevelPct / 100,
    };
    const r = await fetch(`${API}/api/analyze`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`analyze failed: ${r.status}`);
    return r.json();
  }

  async function fetchServerStock(): Promise<StockResponse | null> {
    return fetch(`${API}/api/stock/${encodeURIComponent(skuId)}`, {
      credentials: "include",
    })
      .then<StockResponse | null>((r) => (r.ok ? r.json() : null))
      .catch<StockResponse | null>(() => null);
  }

  async function saveServerStock(stock: number): Promise<StockResponse | null> {
    return fetch(`${API}/api/stock/${encodeURIComponent(skuId)}`, {
      method: "PUT",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quantity_on_hand: stock }),
    })
      .then<StockResponse | null>((r) => (r.ok ? r.json() : null))
      .catch<StockResponse | null>(() => null);
  }

  useEffect(() => {
    async function load() {
      try {
        const skuRes = await fetch(`${API}/api/skus/details`, { credentials: "include" }).then((r) => r.json());
        const allSkus: SkuDetail[] = skuRes.skus || [];
        const idx = allSkus.findIndex((s) => s.id === skuId);
        const info = idx >= 0 ? allSkus[idx] : null;
        setSkuInfo(info);
        setSkuIndex(idx >= 0 ? idx : 0);

        const demoStock = info ? computeDemoStock(info.avg_demand, idx) : 50;
        const serverStock = await fetchServerStock();
        const stock = serverStock?.quantity_on_hand ?? getStockForSku(skuId, demoStock);
        const origin = serverStock ? "server" : getStockOrigin(skuId);
        setCurrentStock(stock);
        setStockOrigin(origin);
        setStockDraft(String(stock));

        const [analyzeRes, historyRes] = await Promise.all([
          fetchAnalysis(stock, leadTimeDays, serviceLevelPct),
          fetch(`${API}/api/skus/${encodeURIComponent(skuId)}/history?days=${HISTORY_DAYS}`, {
            credentials: "include",
          })
            .then<HistoryResponse>((r) =>
              r.ok ? r.json() : { sku: skuId, available: false, history: [] },
            )
            .catch<HistoryResponse>(() => ({ sku: skuId, available: false, history: [] })),
        ]);

        setAnalysis(analyzeRes);
        setHistory(historyRes.history ?? []);
        setHistoryAvailable(Boolean(historyRes.available) && (historyRes.history?.length ?? 0) > 0);
        setHistorySummary(historyRes.summary ?? null);
      } catch (e) {
        console.error(e);
        setError("Could not connect to the backend. Make sure the API server is running.");
      }
      setLoading(false);
    }
    load();
    // We deliberately don't refetch on assumption change here — the
    // Assumptions panel owns that via ``applyAssumptions`` below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [skuId]);

  async function applyAssumptions(nextStock: number, nextLead: number, nextSvcPct: number) {
    if (!skuInfo && !currentStock) return;
    setReAnalyzing(true);
    try {
      const next = await fetchAnalysis(nextStock, nextLead, nextSvcPct);
      setAnalysis(next);
      setCurrentStock(nextStock);
      setLeadTimeDays(nextLead);
      setServiceLevelPct(nextSvcPct);
    } catch (e) {
      console.error(e);
      setError("Could not re-run analysis with the new assumptions.");
    } finally {
      setReAnalyzing(false);
    }
  }

  async function commitStockDraft() {
    const parsed = Number(stockDraft);
    if (!Number.isFinite(parsed) || parsed < 0) {
      setStockDraft(String(currentStock));
      return;
    }
    const next = Math.round(parsed);
    if (next === currentStock && stockOrigin !== "demo") return;
    const serverStock = await saveServerStock(next);
    if (serverStock) {
      setStockOrigin("server");
    } else {
      setStockForSku(skuId, next);
      setStockOrigin("user");
    }
    void applyAssumptions(next, leadTimeDays, serviceLevelPct);
  }

  function resetStockToDemo() {
    clearStockForSku(skuId);
    const demo = skuInfo ? computeDemoStock(skuInfo.avg_demand, skuIndex) : 50;
    setStockDraft(String(demo));
    setStockOrigin("demo");
    void applyAssumptions(demo, leadTimeDays, serviceLevelPct);
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const stock = currentStock;
  const name = skuInfo?.name || `SKU ${skuId}`;

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-gray-800 bg-gray-950/80 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-1.5 text-gray-400 hover:text-white transition-colors text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
            >
              <ArrowLeft className="w-4 h-4" />
              Dashboard
            </Link>
            <div className="h-5 w-px bg-gray-800" />
            <div>
              <h1 className="font-bold text-sm">{name}</h1>
              <span className="text-xs text-gray-500 font-mono">{skuId}</span>
            </div>
          </div>
          {analysis && <RiskBadge risk={analysis.risk} />}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-6 py-4 text-sm text-red-400 flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 ml-4 font-bold">&times;</button>
          </div>
        )}
        {analysis?.demand_source === "synthetic" && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl px-5 py-3 text-sm text-amber-200 flex items-start gap-3">
            <AlertTriangle className="w-4 h-4 mt-0.5 text-amber-300 shrink-0" />
            <div>
              <p className="font-semibold text-amber-200">Demo data for unknown SKU</p>
              <p className="text-xs text-amber-200/80 mt-1">
                This SKU is not in the processed dataset. Demand, forecast, and reorder values below are generated
                from a synthetic demand series for demonstration — they do not represent real operational data.
              </p>
            </div>
          </div>
        )}

        {/* Analysis summary — the one-card "what matters first" block.
            Recommendation dominates; pattern/forecast/input-data are the
            supporting context underneath. */}
        {analysis && (
          <section
            className="rounded-xl border-2 overflow-hidden"
            style={{
              borderColor: `${analysis.risk_color}40`,
              background: `linear-gradient(180deg, ${analysis.risk_color}0c, transparent)`,
            }}
          >
            <div className="px-6 py-5 flex flex-col md:flex-row md:items-center md:justify-between gap-5">
              <div className="flex items-baseline gap-4 flex-wrap">
                <div>
                  <p
                    className="text-[10px] font-semibold uppercase tracking-[0.18em]"
                    style={{ color: analysis.risk_color }}
                  >
                    {analysis.action === "PURCHASE" ? "Recommended action" : "Current status"}
                  </p>
                  <p className="mt-1 flex items-baseline gap-3">
                    {analysis.recommended_order > 0 ? (
                      <>
                        <span className="text-3xl md:text-[36px] font-bold text-white tracking-tight tabular-nums">
                          Order {analysis.recommended_order}
                        </span>
                        <span className="text-sm text-gray-400">units</span>
                        {analysis.decision?.constraints?.max_order_cap_applied && (
                          <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-amber-300 border border-amber-500/30 bg-amber-500/10 rounded-full px-2 py-1">
                            Max order cap
                          </span>
                        )}
                      </>
                    ) : (
                      <span className="text-3xl md:text-[36px] font-bold text-white tracking-tight">
                        No action needed
                      </span>
                    )}
                  </p>
                </div>
              </div>
              <div className="md:text-right">
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em]">
                  Stock-out risk
                </p>
                <p
                  className="text-lg font-semibold mt-1"
                  style={{ color: analysis.risk_color }}
                >
                  {analysis.risk}
                </p>
              </div>
            </div>
            <div className="px-6 py-3 border-t border-gray-800/60 bg-gray-900/40 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-gray-500 uppercase tracking-wider text-[10px]">Pattern</span>
                <span className="text-purple-300 font-medium">{formatDemandPattern(analysis.demand_pattern)}</span>
              </div>
              <div className="h-3 w-px bg-gray-800 hidden sm:block" />
              <div className="flex items-center gap-2">
                <span className="text-gray-500 uppercase tracking-wider text-[10px]">Forecast</span>
                <span className="text-cyan-300 font-medium">{formatForecastMethod(analysis.forecast_method)}</span>
                <DataSourceBadge kind={forecastSourceKind(analysis.forecast_source)} />
              </div>
              <div className="h-3 w-px bg-gray-800 hidden sm:block" />
              <div className="flex items-center gap-2">
                <span className="text-gray-500 uppercase tracking-wider text-[10px]">Input data</span>
                <DataSourceBadge kind={demandSourceKind(analysis.demand_source)} />
              </div>
            </div>
          </section>
        )}

        {/* Planning assumptions — compact controls that re-run the analysis
            so the user can see how recommendations change when lead time or
            service level changes. */}
        {analysis && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <SectionHeader
              compact
              eyebrow="Interactive"
              title="Planning assumptions"
              subtitle="Change a slider or type a stock value. Releasing the slider or pressing Enter re-runs the analysis."
              right={
                reAnalyzing ? (
                  <span className="text-[11px] text-gray-400 inline-flex items-center gap-2">
                    <span className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    Re-analyzing…
                  </span>
                ) : (
                  <span className="text-[11px] text-gray-600">Defaults: 7 days · 95%</span>
                )
              }
            />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mt-4">
              <label className="block">
                <div className="flex items-baseline justify-between">
                  <span className="text-[11px] text-gray-500 uppercase tracking-wider">
                    Lead time
                  </span>
                  <span className="text-sm text-white font-mono tabular-nums">
                    {leadTimeDays}
                    <span className="text-[10px] text-gray-500 ml-1">days</span>
                    {leadTimeDays !== 7 && (
                      <span className="text-[10px] text-amber-300 ml-2">
                        {leadTimeDays > 7 ? "+" : ""}
                        {leadTimeDays - 7} vs default
                      </span>
                    )}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={30}
                  step={1}
                  value={leadTimeDays}
                  disabled={reAnalyzing}
                  aria-label="Lead time in days"
                  onChange={(e) => setLeadTimeDays(Number(e.target.value))}
                  onMouseUp={() => applyAssumptions(currentStock, leadTimeDays, serviceLevelPct)}
                  onTouchEnd={() => applyAssumptions(currentStock, leadTimeDays, serviceLevelPct)}
                  onKeyUp={(e) => {
                    if (e.key === "Enter" || e.key === "ArrowLeft" || e.key === "ArrowRight") {
                      applyAssumptions(currentStock, leadTimeDays, serviceLevelPct);
                    }
                  }}
                  className="mt-2 w-full accent-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                />
                <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                  <span>1d</span>
                  <span className="text-gray-500">7d default</span>
                  <span>30d</span>
                </div>
              </label>
              <label className="block">
                <div className="flex items-baseline justify-between">
                  <span className="text-[11px] text-gray-500 uppercase tracking-wider">
                    Service level
                  </span>
                  <span className="text-sm text-white font-mono tabular-nums">
                    {serviceLevelPct}%
                    {serviceLevelPct !== 95 && (
                      <span className="text-[10px] text-amber-300 ml-2">
                        {serviceLevelPct > 95 ? "+" : ""}
                        {serviceLevelPct - 95}pp vs default
                      </span>
                    )}
                  </span>
                </div>
                <input
                  type="range"
                  min={51}
                  max={99}
                  step={1}
                  value={serviceLevelPct}
                  disabled={reAnalyzing}
                  aria-label="Target service level percentage"
                  onChange={(e) => setServiceLevelPct(Number(e.target.value))}
                  onMouseUp={() => applyAssumptions(currentStock, leadTimeDays, serviceLevelPct)}
                  onTouchEnd={() => applyAssumptions(currentStock, leadTimeDays, serviceLevelPct)}
                  onKeyUp={(e) => {
                    if (e.key === "Enter" || e.key === "ArrowLeft" || e.key === "ArrowRight") {
                      applyAssumptions(currentStock, leadTimeDays, serviceLevelPct);
                    }
                  }}
                  className="mt-2 w-full accent-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                />
                <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                  <span>51%</span>
                  <span className="text-gray-500">95% default</span>
                  <span>99%</span>
                </div>
              </label>
              <label className="block">
                <div className="flex items-baseline justify-between">
                  <span className="text-[11px] text-gray-500 uppercase tracking-wider">
                    Current stock
                  </span>
                  <span
                    className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded border ${
                      stockOrigin !== "demo"
                        ? "text-emerald-300 bg-emerald-500/10 border-emerald-500/30"
                        : "text-gray-400 bg-gray-500/10 border-gray-500/30"
                    }`}
                    title={
                      stockOrigin === "user"
                        ? "Saved locally as a demo fallback."
                        : stockOrigin === "server"
                          ? "Stored by the backend stock API."
                          : "Demo value derived from this SKU's average demand."
                    }
                  >
                    {stockOrigin === "server"
                      ? "Server stored"
                      : stockOrigin === "user"
                        ? "Local fallback"
                        : "Demo value"}
                  </span>
                </div>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={stockDraft}
                  disabled={reAnalyzing}
                  aria-label="Current stock on hand (units)"
                  onChange={(e) => setStockDraft(e.target.value)}
                  onBlur={commitStockDraft}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      (e.target as HTMLInputElement).blur();
                    } else if (e.key === "Escape") {
                      e.preventDefault();
                      setStockDraft(String(currentStock));
                      (e.target as HTMLInputElement).blur();
                    }
                  }}
                  className="mt-2 w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white font-mono tabular-nums focus:outline-none focus:border-blue-500 focus-visible:ring-2 focus-visible:ring-blue-500"
                />
                <div className="flex justify-between items-center text-[10px] text-gray-600 mt-1">
                  <span>units on hand</span>
                  {stockOrigin === "user" ? (
                    <button
                      type="button"
                      onClick={resetStockToDemo}
                      disabled={reAnalyzing}
                      className="text-gray-500 hover:text-gray-300 underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                    >
                      Reset to demo
                    </button>
                  ) : (
                    <span className="text-gray-500">press Enter to apply</span>
                  )}
                </div>
              </label>
            </div>
          </div>
        )}

        {/* Why this forecast path — deterministic, template-generated
            explanations tied to the real routing and stock inputs. */}
        {analysis?.explanation && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <SectionHeader
              compact
              eyebrow="Reasoning"
              title="Why this forecast path"
              subtitle="Deterministic explanations composed from the real inputs — no LLM, no fabricated confidence score."
            />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs">
              <div>
                <p className="text-[11px] text-gray-500 uppercase tracking-wider mb-1">
                  Classification
                </p>
                <p className="text-gray-300 leading-relaxed">
                  {analysis.explanation.classification_reason}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-gray-500 uppercase tracking-wider mb-1">
                  Method choice
                </p>
                <p className="text-gray-300 leading-relaxed">
                  {analysis.explanation.method_reason}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-gray-500 uppercase tracking-wider mb-1">
                  Risk reasoning
                </p>
                <p className="text-gray-300 leading-relaxed">
                  {analysis.explanation.risk_reason}
                </p>
              </div>
            </div>
            <p className="text-[11px] text-gray-500 border-t border-gray-800 pt-3 mt-5 leading-relaxed">
              {analysis.explanation.confidence_note}
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left: Recorded sales history (3 cols) — the core "actual past
              data" surface. Deliberately titled and labeled so nothing here
              can be mistaken for a forecast. Forecast values appear only as
              dashed reference lines overlaid on top. */}
          <div className="lg:col-span-3 bg-gray-900 border border-gray-800 rounded-xl p-6">
            <SectionHeader
              compact
              eyebrow="Actuals"
              title={
                <span className="inline-flex items-center gap-2">
                  Recorded sales history
                  {historyAvailable && (
                    <span className="text-xs font-normal text-gray-500">
                      · last {history.length} days
                    </span>
                  )}
                  {historyAvailable && (
                    <DataSourceBadge kind="recorded" />
                  )}
                </span>
              }
              subtitle="Each blue bar is the real number of units sold on that day, pulled directly from the processed sales dataset. Dashed lines are historical demand reference levels, not future forecast lines."
              right={
                analysis && (
                  <div className="flex items-center gap-3 text-[11px] text-gray-500 flex-wrap">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-sm bg-blue-500" />
                      <span className="text-gray-300">Actual units sold</span>
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-3 border-t border-dashed border-green-500" /> Historical avg
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-3 border-t border-dashed border-red-500" /> Historical P90
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-3 h-px bg-yellow-500" /> Current stock
                    </span>
                  </div>
                )
              }
            />
            {analysis && historyAvailable ? (
              <>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={history} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                    <XAxis
                      dataKey="date"
                      tick={{ fill: "#6b7280", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(value: string) => value.slice(5)}
                      minTickGap={24}
                    />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                    <Tooltip
                      contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
                      labelStyle={{ color: "#9ca3af" }}
                      formatter={(value) => [`${value as number} units`, "Actual units sold"]}
                      labelFormatter={(label) => `Date: ${label as string}  (recorded sale)`}
                    />
                    <Bar dataKey="demand" name="Actual units sold" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <ReferenceLine
                      y={analysis.forecast.p50}
                      stroke="#22c55e"
                      strokeDasharray="6 3"
                      label={{ value: `Historical avg: ${analysis.forecast.p50}`, position: "right", fill: "#22c55e", fontSize: 11 }}
                    />
                    <ReferenceLine
                      y={analysis.forecast.p90}
                      stroke="#ef4444"
                      strokeDasharray="6 3"
                      label={{ value: `Historical P90: ${analysis.forecast.p90}`, position: "right", fill: "#ef4444", fontSize: 11 }}
                    />
                    <ReferenceLine
                      y={stock}
                      stroke="#eab308"
                      label={{ value: `Stock: ${stock}`, position: "right", fill: "#eab308", fontSize: 11 }}
                    />
                  </BarChart>
                </ResponsiveContainer>

                {/* Summary strip — quick-read aggregates over the recorded window. */}
                {historySummary && (
                  <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div className="bg-gray-950/60 border border-gray-800 rounded-lg px-3 py-2">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider">Total sold</p>
                      <p className="text-sm font-semibold text-white tabular-nums mt-0.5">
                        {Math.round(historySummary.total_units_sold).toLocaleString()}
                        <span className="text-[10px] text-gray-500 font-normal ml-1">units</span>
                      </p>
                    </div>
                    <div className="bg-gray-950/60 border border-gray-800 rounded-lg px-3 py-2">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider">Daily avg</p>
                      <p className="text-sm font-semibold text-white tabular-nums mt-0.5">
                        {historySummary.mean_units_per_day.toFixed(1)}
                        <span className="text-[10px] text-gray-500 font-normal ml-1">/day</span>
                      </p>
                    </div>
                    <div className="bg-gray-950/60 border border-gray-800 rounded-lg px-3 py-2">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider">Peak day</p>
                      <p className="text-sm font-semibold text-white tabular-nums mt-0.5">
                        {Math.round(historySummary.peak_units_in_one_day).toLocaleString()}
                        <span className="text-[10px] text-gray-500 font-normal ml-1">units</span>
                      </p>
                    </div>
                    <div className="bg-gray-950/60 border border-gray-800 rounded-lg px-3 py-2">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider">Zero-sale days</p>
                      <p className="text-sm font-semibold text-white tabular-nums mt-0.5">
                        {historySummary.days_with_zero_sales}
                        <span className="text-[10px] text-gray-500 font-normal ml-1">
                          /{historySummary.window_days_returned}
                        </span>
                      </p>
                    </div>
                  </div>
                )}

                {/* Recent recorded sales — lets a reviewer verify dated actuals
                    directly, not just inferred from bar height. */}
                <div className="mt-5 border-t border-gray-800 pt-4">
                  <div className="flex items-baseline justify-between mb-2">
                    <p className="text-[11px] font-semibold text-gray-300 uppercase tracking-wider">
                      Recent recorded sales
                    </p>
                    <span className="text-[10px] text-gray-500">
                      exact daily values — most recent last
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-left text-gray-500 border-b border-gray-800">
                          <th className="py-1.5 pr-4 font-medium uppercase tracking-wider text-[10px]">Date</th>
                          <th className="py-1.5 pr-4 font-medium uppercase tracking-wider text-[10px] text-right">Units sold</th>
                          <th className="py-1.5 font-medium uppercase tracking-wider text-[10px]">Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {history.slice(-7).reverse().map((row) => {
                          const value = row.units_sold ?? row.demand;
                          return (
                            <tr key={row.date} className="border-b border-gray-800/50 last:border-b-0">
                              <td className="py-1.5 pr-4 font-mono text-gray-300">{row.date}</td>
                              <td className="py-1.5 pr-4 text-right tabular-nums text-white font-medium">
                                {Math.round(value).toLocaleString()}
                              </td>
                              <td className="py-1.5 text-gray-500 text-[11px]">
                                {value === 0 ? "no sales recorded" : "recorded actual"}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-3 text-[10px] text-gray-500 leading-relaxed">
                    These are the exact values returned by
                    {" "}<code className="px-1 py-0.5 rounded bg-gray-800 text-gray-300 text-[10px]">GET /api/skus/{skuId}/history</code>{" "}
                    with <code className="px-1 py-0.5 rounded bg-gray-800 text-gray-300 text-[10px]">series_type=recorded_history</code>.
                    Forecasted future values never appear in this list.
                  </p>
                </div>
              </>
            ) : (
              <EmptyState
                title="No recorded sales history"
                hint={
                  <>
                    The backend has no recorded sales for this SKU in the processed dataset, so
                    there are no past actuals to show. The recommendation on the right still uses
                    the backend&apos;s forecast logic (labeled honestly as <span className="text-amber-300">synthetic</span>
                    {" "}or <span className="text-amber-300">rule-based fallback</span>) — only the
                    actuals chart is hidden.
                  </>
                }
                className="h-[280px]"
              />
            )}
          </div>

          {/* Right: Decision + Stats (2 cols) */}
          <div className="lg:col-span-2 space-y-4">
            {/* Decision rationale — complements the top hero with the full
                human-readable "why". */}
            {analysis && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                <SectionHeader
                  compact
                  eyebrow="Decision"
                  title="Why this quantity"
                />
                {analysis.decision?.why ? (
                  <div className="space-y-4">
                    <p className="text-xs text-gray-300 leading-relaxed">
                      {analysis.decision.why}
                    </p>
                    {analysis.decision.constraints?.constrained && (
                      <div className="rounded-lg border border-gray-800 bg-gray-950/60 p-3 space-y-2">
                        <div className="flex items-center justify-between gap-4 text-xs">
                          <span className="text-gray-500">Raw requirement</span>
                          <span className="font-mono tabular-nums text-white">
                            {formatUnits(analysis.decision.constraints.raw_order_quantity)}
                          </span>
                        </div>
                        {analysis.decision.constraints.max_order_cap_applied && (
                          <div className="flex items-center justify-between gap-4 text-xs">
                            <span className="text-gray-500">Supplier max order</span>
                            <span className="font-mono tabular-nums text-amber-300">
                              {formatUnits(analysis.decision.constraints.max_order_quantity)}
                            </span>
                          </div>
                        )}
                        <div className="flex items-center justify-between gap-4 text-xs">
                          <span className="text-gray-500">Final recommendation</span>
                          <span className="font-mono tabular-nums text-green-300">
                            {formatUnits(analysis.decision.constraints.final_order_quantity)}
                          </span>
                        </div>
                        {analysis.decision.constraints.remaining_gap_after_order > 0 && (
                          <div className="flex items-center justify-between gap-4 border-t border-gray-800 pt-2 text-xs">
                            <span className="text-gray-500">Remaining gap after this order</span>
                            <span className="font-mono tabular-nums text-red-300">
                              {formatUnits(analysis.decision.constraints.remaining_gap_after_order)}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-gray-500">
                    {analysis.action === "PURCHASE"
                      ? `Order ${analysis.recommended_order} units to bring stock above the reorder point.`
                      : "Current stock already covers the reorder point — no action needed."}
                  </p>
                )}
              </div>
            )}

            {/* Stats Grid */}
            {analysis && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                <SectionHeader compact eyebrow="Metrics" title="Key metrics" />
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-4">
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="text-[11px] text-gray-500 uppercase tracking-wider">
                        Current stock
                      </p>
                      <DataSourceBadge
                        kind={
                          stockOrigin === "server"
                            ? "server_stored"
                            : stockOrigin === "user"
                              ? "user_stored"
                              : "demo_value"
                        }
                      />
                    </div>
                    <p className="text-2xl font-bold tabular-nums mt-1">{stock}</p>
                  </div>
                  <div>
                    <p className="text-[11px] text-gray-500 uppercase tracking-wider">
                      Historical avg
                    </p>
                    <p className="text-2xl font-bold text-green-400 tabular-nums mt-1">
                      {analysis.forecast.p50}
                    </p>
                  </div>
                  <div>
                    <p className="text-[11px] text-gray-500 uppercase tracking-wider">
                      Historical P90
                    </p>
                    <p className="text-2xl font-bold text-red-400 tabular-nums mt-1">
                      {analysis.forecast.p90}
                    </p>
                  </div>
                  <div>
                    <p
                      className="text-[11px] text-gray-500 uppercase tracking-wider"
                      title="How much current stock undershoots the P90 demand scenario."
                    >
                      Shortfall vs P90
                    </p>
                    <p className="text-2xl font-bold text-yellow-400 tabular-nums mt-1">
                      {Math.max(0, Math.round(analysis.forecast.p90 - stock))}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Model & Method provenance — always truthful: for statistical
                and rule-based paths we show the method name, not a fake
                model identifier. */}
            {analysis?.model_info && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                <SectionHeader compact eyebrow="Provenance" title="Model & method" />
                <div className="space-y-2.5">
                <InfoRow label="Name" value={analysis.model_info.model_name} />
                <InfoRow
                  label="Type"
                  value={analysis.model_info.model_type.replace(/_/g, " ")}
                  emphasis={
                    analysis.model_info.model_type === "rule_based_fallback" ||
                    analysis.model_info.model_type === "none"
                      ? "warning"
                      : "default"
                  }
                />
                <InfoRow
                  label="Artifact"
                  value={analysis.model_info.artifact_available ? "loaded" : "not loaded"}
                  emphasis={analysis.model_info.artifact_available ? "default" : "warning"}
                />
                {analysis.model_info.trained_at && (
                  <InfoRow
                    label="Trained"
                    value={formatShortDate(analysis.model_info.trained_at)}
                    hint={analysis.model_info.trained_at}
                  />
                )}
                {typeof analysis.model_info.feature_count === "number" && (
                  <InfoRow
                    label="Features"
                    value={analysis.model_info.feature_count}
                  />
                )}
                {analysis.model_info.dataset && (
                  <InfoRow
                    label="Dataset"
                    value={analysis.model_info.dataset}
                    emphasis="muted"
                  />
                )}
                <InfoRow
                  label="Evaluation"
                  value={
                    analysis.model_info.evaluation_available
                      ? `available (${formatShortDate(
                          analysis.model_info.evaluation_generated_at ?? undefined
                        )})`
                      : "not generated"
                  }
                  emphasis={analysis.model_info.evaluation_available ? "muted" : "warning"}
                />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* How This Decision Was Made */}
        {analysis && (
          <section className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <SectionHeader
              eyebrow="Pipeline"
              title="How this decision was made"
              subtitle="Four-step breakdown of the pipeline that produced the recommendation for this specific SKU."
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
              {/* Step 1: Classify */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-9 h-9 rounded-lg bg-purple-600/20 flex items-center justify-center">
                    <Tags className="w-4 h-4 text-purple-400" />
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">1. Classify</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Demand pattern:{" "}
                    <span className="text-purple-400 font-medium">{formatDemandPattern(analysis.demand_pattern)}</span>.{" "}
                    {PATTERN_REASONS[analysis.demand_pattern] || ""}
                  </p>
                </div>
              </div>

              {/* Step 2: Forecast */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-9 h-9 rounded-lg bg-cyan-600/20 flex items-center justify-center">
                    <TrendingUp className="w-4 h-4 text-cyan-400" />
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">2. Forecast</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Method: <span className="text-cyan-400 font-medium">{formatForecastMethod(analysis.forecast_method)}</span>.
                    {analysis.decision && (
                      <>
                        {" "}Projected demand over the next{" "}
                        <span className="text-cyan-300 font-medium">
                          {analysis.decision.lead_time_days} days
                        </span>
                        {" = "}
                        <span className="text-cyan-300 font-medium">
                          {analysis.decision.lead_time_demand} units
                        </span>
                        .
                      </>
                    )}
                  </p>
                </div>
              </div>

              {/* Step 3: Safety Stock */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-9 h-9 rounded-lg bg-blue-600/20 flex items-center justify-center">
                    <Shield className="w-4 h-4 text-blue-400" />
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">3. Safety Stock</p>
                  <p className="text-xs text-gray-400 mt-1">
                    {analysis.decision ? (
                      <>
                        Buffer of{" "}
                        <span className="text-blue-300 font-medium">
                          {analysis.decision.safety_stock} units
                        </span>{" "}
                        at{" "}
                        <span className="text-blue-300 font-medium">
                          {Math.round(analysis.decision.service_level * 100)}%
                        </span>{" "}
                        service level ({analysis.decision.safety_stock_method}). Reorder point ={" "}
                        <span className="text-blue-300 font-medium">
                          {analysis.decision.reorder_point}
                        </span>
                        .
                      </>
                    ) : (
                      <>Safety stock sized from forecast error at 95% service level.</>
                    )}
                  </p>
                </div>
              </div>

              {/* Step 4: Decision */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-9 h-9 rounded-lg bg-green-600/20 flex items-center justify-center">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">4. Decision</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Current stock{" "}
                    <span className="text-white font-medium">{stock}</span>{" "}
                    {analysis.decision ? (
                      analysis.recommended_order > 0 ? (
                        <>
                          is{" "}
                          <span className="text-yellow-300 font-medium">
                            {analysis.decision.inventory_gap}
                          </span>{" "}
                          below the reorder point → order{" "}
                          <span className="text-green-400 font-medium">
                            {analysis.recommended_order}
                          </span>{" "}
                          units.
                        </>
                      ) : (
                        <>
                          already covers the reorder point (
                          {analysis.decision.reorder_point}) — no order.
                        </>
                      )
                    ) : analysis.recommended_order > 0 ? (
                      <> → order {analysis.recommended_order} units.</>
                    ) : (
                      <> → no action needed.</>
                    )}
                  </p>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}


