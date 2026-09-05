"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Database,
  Brain,
  BarChart3,
  ShoppingCart,
  ArrowRight,
  TrendingDown,
  Target,
  Package,
  Cpu,
  Search,
  Activity,
} from "lucide-react";
import {
  api,
  computeDemoStock,
  type SkuDetail,
  type SkuAnalysis,
  type KpiData,
  type HealthStatus,
  type RecentAnalysis,
  type StockLevel,
  type ModelMonitoringSnapshot,
  type HistoricalReplayResponse,
} from "@/lib/api";
import {
  DataSourceBadge,
  forecastSourceKind,
} from "@/components/DataSourceBadge";
import { EmptyState } from "@/components/EmptyState";
import { ModelHealthCard } from "@/components/ModelHealthCard";
import { SectionHeader } from "@/components/SectionHeader";
import { env } from "@/lib/env";
import {
  formatDemandPattern,
  formatForecastMethod,
  formatNumber,
  formatRelativeTime,
} from "@/lib/utils";
import { ChevronRight, Pencil } from "lucide-react";
import { getStockForSku, setStockForSku, getStockOrigin } from "@/lib/stock";

const API_URL = env.apiUrl;

// --- Small Components ---

function RiskBadge({ risk }: { risk: string }) {
  const styles: Record<string, string> = {
    HIGH: "bg-red-500/15 text-red-400 border-red-500/30",
    MEDIUM: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30",
    LOW: "bg-green-500/15 text-green-400 border-green-500/30",
  };
  return (
    <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${styles[risk] || styles.LOW}`}>
      {risk}
    </span>
  );
}

function PatternBadge({ pattern }: { pattern: string }) {
  const colors: Record<string, string> = {
    regular: "text-blue-400",
    intermittent: "text-purple-400",
    highly_intermittent: "text-orange-400",
  };
  return <span className={`text-xs ${colors[pattern] || "text-gray-400"}`}>{formatDemandPattern(pattern)}</span>;
}

function SkeletonRow({ index }: { index: number }) {
  const widths = [60, 80, 40, 50, 55, 65, 45];
  return (
    <tr className="border-b border-gray-800/50">
      {widths.map((w, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 bg-gray-800 rounded animate-pulse" style={{ width: `${w + (index * 7 + i * 3) % 20}%` }} />
        </td>
      ))}
    </tr>
  );
}

// --- Pipeline Step ---

function PipelineStep({
  icon: Icon,
  title,
  desc,
  color,
}: {
  icon: React.ElementType;
  title: string;
  desc: string;
  color: string;
}) {
  return (
    <div className="flex flex-col items-center text-center gap-2">
      <div className={`w-12 h-12 rounded-xl ${color} flex items-center justify-center`}>
        <Icon className="w-6 h-6 text-white" />
      </div>
      <p className="text-sm font-semibold text-white">{title}</p>
      <p className="text-xs text-gray-500">{desc}</p>
    </div>
  );
}

// --- KPI Card ---

function KpiCard({
  icon: Icon,
  value,
  label,
  sublabel,
  color,
  tooltip,
}: {
  icon: React.ElementType;
  value: string;
  label: string;
  sublabel: string;
  color: string;
  tooltip?: string;
}) {
  return (
    <div
      className="group bg-gray-900 border border-gray-800 rounded-xl p-5 flex flex-col gap-3 transition-colors hover:border-gray-700"
      title={tooltip}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className={`w-8 h-8 rounded-lg ${color} flex items-center justify-center shrink-0`}>
            <Icon className="w-4 h-4 text-white" />
          </div>
          <span className="text-[11px] text-gray-500 font-medium uppercase tracking-wider truncate">
            {label}
          </span>
        </div>
        {tooltip && (
          <span
            aria-hidden="true"
            className="text-[10px] text-gray-600 group-hover:text-gray-400 transition-colors select-none"
          >
            ⓘ
          </span>
        )}
      </div>
      <p className="text-[28px] leading-none font-bold text-white tracking-tight">{value}</p>
      <p className="text-xs text-gray-500 leading-snug">{sublabel}</p>
    </div>
  );
}

// --- Main Page ---

export default function Dashboard() {
  const router = useRouter();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [kpis, setKpis] = useState<KpiData | null>(null);
  const [skuDetails, setSkuDetails] = useState<SkuDetail[]>([]);
  const [analyses, setAnalyses] = useState<Record<string, SkuAnalysis>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [kpisUnavailable, setKpisUnavailable] = useState(false);
  const [skusUnavailable, setSkusUnavailable] = useState(false);
  const [recentAnalyses, setRecentAnalyses] = useState<RecentAnalysis[]>([]);
  const [serverStock, setServerStock] = useState<Record<string, StockLevel>>({});
  const [modelMonitoring, setModelMonitoring] = useState<ModelMonitoringSnapshot | null>(null);
  const [modelMonitoringLoading, setModelMonitoringLoading] = useState(true);
  const [modelMonitoringError, setModelMonitoringError] = useState(false);
  const [modelMonitoringReplay, setModelMonitoringReplay] = useState<HistoricalReplayResponse | null>(null);
  const [activeFilter, setActiveFilter] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  // Inline stock editing — null when nothing is being edited, otherwise the
  // SKU id and the draft string.
  const [editingStock, setEditingStock] = useState<{ sku: string; draft: string } | null>(null);
  const [stockVersion, setStockVersion] = useState(0); // force re-render after fallback stock edits

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function fetchData() {
      try {
        api
          .getModelMonitoring()
          .then((snapshot) => {
            if (!cancelled) {
              setModelMonitoring(snapshot);
              setModelMonitoringError(false);
            }
          })
          .catch(() => {
            if (!cancelled) {
              setModelMonitoring(null);
              setModelMonitoringError(true);
            }
          })
          .finally(() => {
            if (!cancelled) setModelMonitoringLoading(false);
          });

        // Historical replay is independent of live monitoring: it's a small,
        // pre-generated read-only artifact, never live production evidence.
        api
          .getModelMonitoringReplay()
          .then((replay) => {
            if (!cancelled) setModelMonitoringReplay(replay);
          })
          .catch(() => {
            if (!cancelled) setModelMonitoringReplay(null);
          });

        const [healthRes, kpiRes, skuRes, stockRes] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getKpis().catch(() => null),
          api.getSkuDetails().catch(() => null),
          api.getStockLevels().catch(() => null),
        ]);

        if (cancelled) return;
        setHealth(healthRes);
        if (kpiRes) {
          setKpis(kpiRes);
          setKpisUnavailable(false);
        } else {
          setKpisUnavailable(true);
        }
        if (skuRes) {
          setSkuDetails(skuRes.skus || []);
          setSkusUnavailable(false);
        } else {
          setSkuDetails([]);
          setSkusUnavailable(true);
        }
        const stockBySku = Object.fromEntries(
          (stockRes?.items || []).map((item) => [item.sku, item]),
        ) as Record<string, StockLevel>;
        setServerStock(stockBySku);
        setLoading(false);

        const skus = skuRes?.skus || [];
        const batchSize = 5;
        for (let i = 0; i < skus.length; i += batchSize) {
          if (cancelled) return;
          const batch = skus.slice(i, i + batchSize);
          const results = await Promise.all(
            batch.map((sku, batchIdx) => {
              const globalIdx = i + batchIdx;
              const demo = computeDemoStock(sku.avg_demand, globalIdx);
              const stock = stockBySku[sku.id]?.quantity_on_hand ?? getStockForSku(sku.id, demo);
              return api.analyzeSku(sku.id, stock).catch(() => null);
            })
          );
          if (cancelled) return;
          setAnalyses((prev) => {
            const updated = { ...prev };
            results.forEach((r) => {
              if (r && r.sku) updated[r.sku] = r;
            });
            return updated;
          });
        }

        // Pull the most-recent persisted analyses after the batch writes
        // have finished — this shows a review that outputs are not transient.
        const recent = await api.getRecentAnalyses(10).catch(() => null);
        if (!cancelled && recent?.available) {
          setRecentAnalyses(recent.items);
        }
      } catch (e) {
        if (cancelled) return;
        console.error("Failed to fetch data:", e);
        setError("Could not connect to the backend. Make sure the API server is running on " + API_URL);
        setModelMonitoringLoading(false);
        setModelMonitoringError(true);
        setLoading(false);
      }
    }
    fetchData();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  // Filter and search
  const filteredSkus = skuDetails.filter((sku) => {
    const analysis = analyses[sku.id];
    const matchesFilter = activeFilter === "ALL" || analysis?.risk === activeFilter;
    const matchesSearch =
      searchQuery === "" ||
      sku.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      sku.name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const filters = ["ALL", "HIGH", "MEDIUM", "LOW"];
  const filterCounts = {
    ALL: skuDetails.length,
    HIGH: Object.values(analyses).filter((a) => a.risk === "HIGH").length,
    MEDIUM: Object.values(analyses).filter((a) => a.risk === "MEDIUM").length,
    LOW: Object.values(analyses).filter((a) => a.risk === "LOW").length,
  };

  const riskBorder: Record<string, string> = {
    HIGH: "border-l-red-500/70",
    MEDIUM: "border-l-yellow-500/70",
    LOW: "border-l-green-500/70",
  };

  async function commitStockEdit(sku: string, raw: string) {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
      setEditingStock(null);
      return;
    }
    const next = Math.round(parsed);
    const updated = await api.updateStockForSku(sku, next).catch(() => null);
    if (updated) {
      setServerStock((prev) => ({ ...prev, [sku]: updated }));
    } else {
      setStockForSku(sku, next);
      setStockVersion((v) => v + 1);
    }
    setEditingStock(null);
    const refreshed = await api.analyzeSku(sku, next).catch(() => null);
    if (refreshed) {
      setAnalyses((prev) => ({ ...prev, [sku]: refreshed }));
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-gray-800 bg-gray-950/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-sm">
              S
            </div>
            <span className="font-bold text-lg">SupplySync AI</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Activity className={`w-4 h-4 ${health?.status === "online" ? "text-green-500" : "text-gray-600"}`} />
            <span className="text-gray-400">
              {health?.status === "online" ? "System Online" : "Connecting…"}
            </span>
            {health?.model_loaded && (
              <span className="text-xs bg-green-500/10 text-green-400 border border-green-500/20 px-2 py-0.5 rounded-full ml-2">
                Model Loaded
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10 space-y-10">
        {/* Hero */}
        <section className="space-y-4 text-center py-2">
          <p className="text-[10px] font-semibold text-blue-400 uppercase tracking-[0.18em]">
            Inventory decision system
          </p>
          <h1 className="text-4xl md:text-[40px] font-bold tracking-tight text-white">
            Forecast-driven reorder recommendations
          </h1>
          <p className="text-base text-gray-400 max-w-2xl mx-auto leading-relaxed">
            ML-powered inventory optimization across{" "}
            <span className="text-white font-semibold">4,900+ SKUs</span> from{" "}
            <span className="text-white font-semibold">1M+ retail transactions</span> —
            with explicit provenance on every recommendation.
          </p>
          <div className="flex items-center justify-center gap-2 flex-wrap pt-1 text-[11px]">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${
                health?.status === "online"
                  ? "bg-green-500/10 text-green-400 border-green-500/30"
                  : "bg-gray-500/10 text-gray-500 border-gray-500/30"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  health?.status === "online" ? "bg-green-400" : "bg-gray-500"
                }`}
              />
              {health?.status === "online" ? "Backend online" : "Connecting…"}
            </span>
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${
                health?.model_loaded
                  ? "bg-cyan-500/10 text-cyan-300 border-cyan-500/30"
                  : "bg-amber-500/10 text-amber-300 border-amber-500/30"
              }`}
              title={
                health?.model_loaded
                  ? "Trained LightGBM artifact loaded — regular-demand SKUs use the ML forecast path."
                  : "Model artifact not loaded — regular-demand SKUs fall back to a rule-based average."
              }
            >
              {health?.model_loaded ? "ML model loaded" : "Fallback mode"}
            </span>
          </div>
        </section>

        {/* Error Banner */}
        {error && (
          <div role="alert" className="bg-red-500/10 border border-red-500/30 rounded-xl px-6 py-4 text-sm text-red-400 flex items-center justify-between">
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              aria-label="Dismiss error"
              className="text-red-400 hover:text-red-300 ml-4 font-bold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 rounded"
            >
              &times;
            </button>
          </div>
        )}

        {/* Pipeline */}
        <section className="flex items-center justify-center gap-4 flex-wrap">
          <PipelineStep icon={Database} title="Ingest" desc="1M+ transactions" color="bg-blue-600" />
          <ArrowRight className="w-5 h-5 text-gray-600 hidden sm:block" aria-hidden="true" />
          <span className="block sm:hidden h-px w-8 bg-gray-700" aria-hidden="true" />
          <PipelineStep icon={Brain} title="Classify" desc="Demand patterns" color="bg-purple-600" />
          <ArrowRight className="w-5 h-5 text-gray-600 hidden sm:block" aria-hidden="true" />
          <span className="block sm:hidden h-px w-8 bg-gray-700" aria-hidden="true" />
          <PipelineStep icon={BarChart3} title="Forecast" desc="LightGBM + Croston" color="bg-cyan-600" />
          <ArrowRight className="w-5 h-5 text-gray-600 hidden sm:block" aria-hidden="true" />
          <span className="block sm:hidden h-px w-8 bg-gray-700" aria-hidden="true" />
          <PipelineStep icon={ShoppingCart} title="Optimize" desc="Reorder decisions" color="bg-emerald-600" />
        </section>

        {/* KPI Cards */}
        <section>
          <SectionHeader
            eyebrow="Performance"
            title="Backtest Performance"
            subtitle={
              "Results from the project backtest. Hover over each metric for its definition."
            }
          />
          {kpisUnavailable && !kpis && null}
        </section>
        {kpisUnavailable && !kpis && (
          <EmptyState
            title="KPIs not yet computed"
            hint="The backtest metrics have not been generated on this deployment. Run the backend setup scripts to produce them."
            tone="warning"
          />
        )}
        <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <KpiCard
            icon={TrendingDown}
            value={kpis ? `${formatNumber(kpis.cost_savings_pct, { maximumFractionDigits: 1 })}%` : "…"}
            label="Cost Savings"
            sublabel="vs naive fixed-threshold policy"
            color="bg-green-600"
            tooltip={
              kpis?.interpretation?.metric_meanings?.cost_savings_pct ??
              "Simulated total cost (holding + stockout) saved vs the naive baseline."
            }
          />
          <KpiCard
            icon={Target}
            value={kpis ? `${formatNumber(kpis.fill_rate * 100, { maximumFractionDigits: 1 })}%` : "…"}
            label="Fill Rate"
            sublabel={`across ${kpis ? formatNumber(kpis.skus_analyzed) : "…"} simulated SKUs`}
            color="bg-blue-600"
            tooltip={
              kpis?.interpretation?.metric_meanings?.fill_rate ??
              "Fraction of demanded units fulfilled under the intelligent policy."
            }
          />
          <KpiCard
            icon={Package}
            value="4,900+"
            label="SKUs in Dataset"
            sublabel="UCI Online Retail II"
            color="bg-purple-600"
            tooltip="Total unique SKUs in the cleaned Online Retail II dataset. The simulated KPIs use the top 10 by total demand."
          />
          <KpiCard
            icon={Cpu}
            value="LightGBM"
            label="Forecast Model"
            sublabel="Trained on the UCI Online Retail II dataset."
            color="bg-orange-600"
            tooltip="LightGBM with lag and calendar features. See README 'Forecast Evaluation' for per-class metrics against naive, seasonal-naive, moving-avg-7, and Croston baselines."
          />
        </section>

        <ModelHealthCard
          snapshot={modelMonitoring}
          loading={modelMonitoringLoading}
          error={modelMonitoringError}
          replay={modelMonitoringReplay}
        />

        {/* SKU Portfolio */}
        <section className="space-y-4">
          <SectionHeader
            eyebrow="Portfolio"
            title="Per-SKU recommendations"
            subtitle="Click any row to open the analysis workspace for that SKU. Each row carries a live provenance badge so you can see which forecasts came from the trained model vs. a statistical or rule-based path."
          />

        {/* Filter Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div
            role="tablist"
            aria-label="Filter SKUs by risk band"
            className="flex gap-1 bg-gray-900 rounded-lg p-1 border border-gray-800"
          >
            {filters.map((f) => (
              <button
                key={f}
                role="tab"
                aria-selected={activeFilter === f}
                aria-label={`Filter: ${f} risk (${filterCounts[f as keyof typeof filterCounts]})`}
                onClick={() => setActiveFilter(f)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                  activeFilter === f
                    ? "bg-gray-800 text-white"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {f}
                <span className="ml-1.5 text-xs text-gray-600">
                  {filterCounts[f as keyof typeof filterCounts]}
                </span>
              </button>
            ))}
          </div>
          <div className="relative w-full sm:w-72">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" aria-hidden="true" />
            <input
              type="text"
              placeholder="Search SKUs…"
              aria-label="Search SKUs by code or product name"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-800 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-gray-600 focus-visible:ring-2 focus-visible:ring-blue-500"
            />
          </div>
        </div>

        {/* SKU Table */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800 text-left">
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">SKU</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Product</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">
                    <span className="inline-flex items-center gap-1.5">
                      Stock
                      <span
                        className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30"
                        title="Stock values use demo defaults until updated. Saved values are stored on the server when available."
                      >
                        EDITABLE
                      </span>
                    </span>
                  </th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Risk</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Pattern</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Method</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider text-right">Order Qty</th>
                </tr>
              </thead>
              <tbody>
                {loading
                  ? Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} index={i} />)
                  : filteredSkus.map((sku) => {
                      const a = analyses[sku.id];
                      const demo = computeDemoStock(sku.avg_demand, skuDetails.indexOf(sku));
                      // stockVersion is read to force re-render after fallback edits.
                      void stockVersion;
                      const server = serverStock[sku.id];
                      const stock = server?.quantity_on_hand ?? getStockForSku(sku.id, demo);
                      const origin = server ? "server" : getStockOrigin(sku.id);
                      const isEditing = editingStock?.sku === sku.id;
                      const borderClass = a ? riskBorder[a.risk] || "border-l-gray-700" : "border-l-gray-800";
                      return (
                        <tr
                          key={sku.id}
                          onClick={() => router.push(`/sku/${sku.id}`)}
                          className={`group border-b border-gray-800/50 border-l-2 ${borderClass} hover:bg-gray-800/30 cursor-pointer transition-colors`}>
                            <td className="px-4 py-3.5">
                              <span className="font-mono text-sm font-semibold text-blue-400">{sku.id}</span>
                            </td>
                            <td className="px-4 py-3.5 max-w-[280px]">
                              <span className="text-sm text-gray-300 truncate block">{sku.name}</span>
                            </td>
                            <td
                              className="px-4 py-3.5"
                              onClick={(e) => {
                                if (!isEditing) {
                                  e.stopPropagation();
                                  setEditingStock({ sku: sku.id, draft: String(stock) });
                                }
                              }}
                            >
                              {isEditing ? (
                                <input
                                  type="number"
                                  min={0}
                                  step={1}
                                  autoFocus
                                  value={editingStock?.draft ?? ""}
                                  aria-label={`Current stock for ${sku.id}`}
                                  onClick={(e) => e.stopPropagation()}
                                  onChange={(e) =>
                                    setEditingStock({ sku: sku.id, draft: e.target.value })
                                  }
                                  onBlur={() =>
                                    commitStockEdit(sku.id, editingStock?.draft ?? String(stock))
                                  }
                                  onKeyDown={(e) => {
                                    e.stopPropagation();
                                    if (e.key === "Enter") {
                                      e.preventDefault();
                                      (e.target as HTMLInputElement).blur();
                                    } else if (e.key === "Escape") {
                                      e.preventDefault();
                                      setEditingStock(null);
                                    }
                                  }}
                                  className="w-20 bg-gray-950 border border-blue-500/60 rounded px-2 py-1 text-sm font-mono tabular-nums text-white focus:outline-none"
                                />
                              ) : (
                                <span
                                  className="inline-flex items-center gap-1.5 text-sm font-medium text-gray-200 tabular-nums hover:text-white cursor-text"
                                  title={
                                    origin === "user"
                                      ? "Saved in this browser because the stock API was unavailable. Click to edit."
                                      : origin === "server"
                                        ? "Stored by the backend stock API. Click to edit."
                                      : "Demo value. Click to override with real stock."
                                  }
                                >
                                  {formatNumber(stock)}
                                  {origin !== "demo" && (
                                    <span className="w-1 h-1 rounded-full bg-emerald-400" aria-hidden="true" />
                                  )}
                                  <Pencil className="w-3 h-3 text-gray-600 group-hover:text-gray-400 transition-colors" aria-hidden="true" />
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-3.5">
                              {a ? <RiskBadge risk={a.risk} /> : <span className="text-xs text-gray-600">…</span>}
                            </td>
                            <td className="px-4 py-3.5">
                              {a ? <PatternBadge pattern={a.demand_pattern} /> : <span className="text-xs text-gray-600">…</span>}
                            </td>
                            <td className="px-4 py-3.5">
                              {a ? (
                                <span className="inline-flex items-center gap-2 flex-wrap">
                                  <span className="text-xs text-gray-400">{formatForecastMethod(a.forecast_method)}</span>
                                  <DataSourceBadge kind={forecastSourceKind(a.forecast_source)} />
                                </span>
                              ) : (
                                <span className="text-xs text-gray-600">…</span>
                              )}
                            </td>
                            <td className="px-4 py-3.5 text-right">
                              <span className="inline-flex items-center gap-2 justify-end w-full">
                                {a && a.recommended_order > 0 ? (
                                  <span className="text-sm font-semibold text-blue-400 tabular-nums">
                                    {formatNumber(a.recommended_order)}
                                  </span>
                                ) : a ? (
                                  <span className="text-sm text-gray-600">—</span>
                                ) : (
                                  <span className="text-xs text-gray-600">…</span>
                                )}
                                <ChevronRight className="w-3.5 h-3.5 text-gray-700 group-hover:text-gray-400 transition-colors" aria-hidden="true" />
                              </span>
                            </td>
                          </tr>
                      );
                    })}
              </tbody>
            </table>
          </div>
          {!loading && filteredSkus.length === 0 && (
            <div className="p-6">
              {skusUnavailable ? (
                <EmptyState
                  title="SKU list unavailable"
                  hint={
                    <>
                      The backend didn&apos;t return the SKU list. If you&apos;re running
                      locally, check that the API is up and that the processed dataset
                      exists — <code className="px-1 py-0.5 rounded bg-gray-800 text-gray-300 text-[11px]">
                        python scripts/bootstrap.py
                      </code>{" "}in the backend generates it.
                    </>
                  }
                  tone="warning"
                />
              ) : (
                <EmptyState title="No SKUs match your filter" hint="Clear the search or change the filter above." />
              )}
            </div>
          )}
          {!loading && filteredSkus.length > 0 && (
            <div className="px-4 py-3 border-t border-gray-800/50 text-[11px] text-gray-500 leading-relaxed">
              Stock values use <span className="text-amber-300 font-medium">demo defaults</span> until updated. Saved
              values are stored on the server, with a browser fallback if the stock API is unavailable. Method badges identify how each recommendation was produced —{" "}
              <span className="text-cyan-300">model</span>, <span className="text-violet-300">statistical</span>, or{" "}
              <span className="text-amber-300">rule-based fallback</span>.
            </div>
          )}
          </div>
        </section>

        {/* Recent analyses — proof that outputs are persisted, not transient. */}
        {recentAnalyses.length > 0 && (
          <section>
            <SectionHeader
              eyebrow="Activity"
              title="Recent analyses"
              subtitle="Every /api/analyze call is persisted for inspection. Click a row to open that SKU's analysis workspace."
              right={
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full border bg-gray-500/10 text-gray-400 border-gray-500/30 tracking-wider">
                  Persisted
                </span>
              }
            />
            <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-left text-gray-500">
                      <th className="px-4 py-2.5 font-medium uppercase tracking-wider">When</th>
                      <th className="px-4 py-2.5 font-medium uppercase tracking-wider">SKU</th>
                      <th className="px-4 py-2.5 font-medium uppercase tracking-wider">Risk</th>
                      <th className="px-4 py-2.5 font-medium uppercase tracking-wider">Method</th>
                      <th className="px-4 py-2.5 font-medium uppercase tracking-wider text-right">Ordered</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentAnalyses.map((row) => (
                      <tr
                        key={row.id}
                        onClick={() => router.push(`/sku/${row.sku}`)}
                        className="group border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
                      >
                        <td className="px-4 py-2.5 text-gray-400" title={new Date(row.created_at).toLocaleString()}>
                          {formatRelativeTime(row.created_at)}
                        </td>
                        <td className="px-4 py-2.5 font-mono text-blue-400">{row.sku}</td>
                        <td className="px-4 py-2.5">
                          {row.risk ? (
                            <RiskBadge risk={row.risk} />
                          ) : (
                            <span className="text-gray-600">—</span>
                          )}
                        </td>
                        <td className="px-4 py-2.5 text-gray-300">{formatForecastMethod(row.forecast_method) || "—"}</td>
                        <td className="px-4 py-2.5 text-right">
                          <span className="inline-flex items-center gap-2 justify-end w-full">
                            {row.recommended_order && row.recommended_order > 0 ? (
                              <span className="text-blue-400 font-semibold tabular-nums">
                                {formatNumber(row.recommended_order)}
                              </span>
                            ) : (
                              <span className="text-gray-600">—</span>
                            )}
                            <ChevronRight className="w-3.5 h-3.5 text-gray-700 group-hover:text-gray-400 transition-colors" />
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="border-t border-gray-800 pt-6 pb-4 text-center">
          <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
            <span className="text-gray-600">Built with</span>
            {["Python", "FastAPI", "LightGBM", "Next.js", "Tailwind CSS", "Recharts"].map((tech) => (
              <span key={tech} className="px-2.5 py-1 border border-gray-800 rounded-full text-gray-400">
                {tech}
              </span>
            ))}
          </div>
        </footer>
      </main>
    </div>
  );
}

