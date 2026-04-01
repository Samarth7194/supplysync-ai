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
  ArrowUpDown,
  Activity,
} from "lucide-react";

const API = "http://localhost:8000";

// Deterministic stock levels to produce a mix of risk levels
function computeDemoStock(avgDemand: number, index: number): number {
  const multipliers = [0.3, 0.8, 2.0]; // HIGH, MEDIUM, LOW
  return Math.max(1, Math.round(avgDemand * multipliers[index % 3]));
}

// --- Types ---

interface SkuDetail {
  id: string;
  name: string;
  avg_demand: number;
  total_demand: number;
}

interface SkuAnalysis {
  sku: string;
  risk: "HIGH" | "MEDIUM" | "LOW";
  risk_color: string;
  forecast: { p50: number; p90: number };
  current_stock: number;
  recommended_order: number;
  action: string;
  demand_pattern: string;
  forecast_method: string;
}

interface KpiData {
  total_cost: number;
  fill_rate: number;
  cost_savings_pct: number;
  holding_cost: number;
  stockout_cost: number;
  naive_total_cost: number;
  intelligent_total_cost: number;
  skus_analyzed: number;
}

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
  return <span className={`text-xs ${colors[pattern] || "text-gray-400"}`}>{pattern}</span>;
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
}: {
  icon: React.ElementType;
  value: string;
  label: string;
  sublabel: string;
  color: string;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className={`w-8 h-8 rounded-lg ${color} flex items-center justify-center`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
        <span className="text-xs text-gray-500 font-medium uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-3xl font-bold text-white">{value}</p>
      <p className="text-xs text-gray-500">{sublabel}</p>
    </div>
  );
}

// --- Main Page ---

export default function Dashboard() {
  const router = useRouter();
  const [health, setHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [kpis, setKpis] = useState<KpiData | null>(null);
  const [skuDetails, setSkuDetails] = useState<SkuDetail[]>([]);
  const [analyses, setAnalyses] = useState<Record<string, SkuAnalysis>>({});
  const [loading, setLoading] = useState(true);
  const [activeFilter, setActiveFilter] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch all data on mount
  useEffect(() => {
    async function fetchData() {
      try {
        const [healthRes, kpiRes, skuRes] = await Promise.all([
          fetch(`${API}/health`).then((r) => r.json()).catch(() => null),
          fetch(`${API}/api/kpis`).then((r) => r.json()).catch(() => null),
          fetch(`${API}/api/skus/details`).then((r) => r.json()).catch(() => ({ skus: [] })),
        ]);

        setHealth(healthRes);
        if (kpiRes && !kpiRes.error) setKpis(kpiRes);
        setSkuDetails(skuRes.skus || []);
        setLoading(false);

        // Analyze each SKU in batches
        const skus: SkuDetail[] = skuRes.skus || [];
        const batchSize = 5;
        for (let i = 0; i < skus.length; i += batchSize) {
          const batch = skus.slice(i, i + batchSize);
          const results = await Promise.all(
            batch.map((sku, batchIdx) => {
              const globalIdx = i + batchIdx;
              const stock = computeDemoStock(sku.avg_demand, globalIdx);
              return fetch(`${API}/api/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sku: sku.id, current_stock: stock }),
              })
                .then((r) => r.json())
                .catch(() => null);
            })
          );
          setAnalyses((prev) => {
            const updated = { ...prev };
            results.forEach((r) => {
              if (r && r.sku) updated[r.sku] = r;
            });
            return updated;
          });
        }
      } catch (e) {
        console.error("Failed to fetch data:", e);
        setLoading(false);
      }
    }
    fetchData();
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
              {health?.status === "online" ? "System Online" : "Connecting..."}
            </span>
            {health?.model_loaded && (
              <span className="text-xs bg-green-500/10 text-green-400 border border-green-500/20 px-2 py-0.5 rounded-full ml-2">
                Model Loaded
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-10">
        {/* Hero */}
        <section className="text-center space-y-3 py-4">
          <h1 className="text-4xl font-bold tracking-tight">
            Inventory Intelligence
          </h1>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            ML-powered inventory optimization analyzing{" "}
            <span className="text-white font-semibold">4,900+ SKUs</span> from{" "}
            <span className="text-white font-semibold">1M+ retail transactions</span>
          </p>
        </section>

        {/* Pipeline */}
        <section className="flex items-center justify-center gap-4 flex-wrap">
          <PipelineStep icon={Database} title="Ingest" desc="1M+ transactions" color="bg-blue-600" />
          <ArrowRight className="w-5 h-5 text-gray-600 hidden sm:block" />
          <PipelineStep icon={Brain} title="Classify" desc="Demand patterns" color="bg-purple-600" />
          <ArrowRight className="w-5 h-5 text-gray-600 hidden sm:block" />
          <PipelineStep icon={BarChart3} title="Forecast" desc="LightGBM + Croston" color="bg-cyan-600" />
          <ArrowRight className="w-5 h-5 text-gray-600 hidden sm:block" />
          <PipelineStep icon={ShoppingCart} title="Optimize" desc="Reorder decisions" color="bg-emerald-600" />
        </section>

        {/* KPI Cards */}
        <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <KpiCard
            icon={TrendingDown}
            value={kpis ? `${kpis.cost_savings_pct}%` : "..."}
            label="Cost Savings"
            sublabel="vs naive fixed-threshold policy"
            color="bg-green-600"
          />
          <KpiCard
            icon={Target}
            value={kpis ? `${(kpis.fill_rate * 100).toFixed(1)}%` : "..."}
            label="Fill Rate"
            sublabel={`across ${kpis?.skus_analyzed || "..."} simulated SKUs`}
            color="bg-blue-600"
          />
          <KpiCard
            icon={Package}
            value="4,900+"
            label="SKUs Analyzed"
            sublabel="UCI Online Retail II dataset"
            color="bg-purple-600"
          />
          <KpiCard
            icon={Cpu}
            value="LightGBM"
            label="Forecast Model"
            sublabel="MAE: 95.62 on test set"
            color="bg-orange-600"
          />
        </section>

        {/* Filter Bar */}
        <section className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex gap-1 bg-gray-900 rounded-lg p-1 border border-gray-800">
            {filters.map((f) => (
              <button
                key={f}
                onClick={() => setActiveFilter(f)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
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
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search SKUs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-800 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-gray-600"
            />
          </div>
        </section>

        {/* SKU Table */}
        <section className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800 text-left">
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">SKU</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Product</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Stock</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Risk</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Pattern</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Method</th>
                  <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider text-right">Order Qty</th>
                </tr>
              </thead>
              <tbody>
                {loading
                  ? Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} index={i} />)
                  : filteredSkus.map((sku, idx) => {
                      const a = analyses[sku.id];
                      const stock = computeDemoStock(sku.avg_demand, skuDetails.indexOf(sku));
                      return (
                        <tr
                          key={sku.id}
                          onClick={() => router.push(`/sku/${sku.id}`)}
                          className="border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer transition-colors">
                            <td className="px-4 py-3">
                              <span className="font-mono text-sm font-semibold text-blue-400">{sku.id}</span>
                            </td>
                            <td className="px-4 py-3">
                              <span className="text-sm text-gray-300">{sku.name}</span>
                            </td>
                            <td className="px-4 py-3">
                              <span className="text-sm font-medium">{stock}</span>
                            </td>
                            <td className="px-4 py-3">
                              {a ? <RiskBadge risk={a.risk} /> : <span className="text-xs text-gray-600">...</span>}
                            </td>
                            <td className="px-4 py-3">
                              {a ? <PatternBadge pattern={a.demand_pattern} /> : <span className="text-xs text-gray-600">...</span>}
                            </td>
                            <td className="px-4 py-3">
                              <span className="text-xs text-gray-400">{a?.forecast_method || "..."}</span>
                            </td>
                            <td className="px-4 py-3 text-right">
                              {a && a.recommended_order > 0 ? (
                                <span className="text-sm font-semibold text-blue-400">{a.recommended_order}</span>
                              ) : a ? (
                                <span className="text-sm text-gray-600">-</span>
                              ) : (
                                <span className="text-xs text-gray-600">...</span>
                              )}
                            </td>
                          </tr>
                      );
                    })}
              </tbody>
            </table>
          </div>
          {!loading && filteredSkus.length === 0 && (
            <div className="text-center py-12 text-gray-500">No SKUs match your filter.</div>
          )}
        </section>

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
