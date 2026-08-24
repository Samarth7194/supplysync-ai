import { env } from "./env";
import { MODEL_MONITORING_ENDPOINT } from "./modelMonitoring";

const API = env.apiUrl;

export interface SkuDetail {
  id: string;
  name: string;
  avg_demand: number;
  total_demand: number;
}

export type DemandSource = "request" | "historical" | "synthetic";
export type ForecastSource =
  | "model_forecast"
  | "statistical_method"
  | "rule_based_estimate"
  | "unavailable";

export type ModelType = "ml" | "statistical_method" | "rule_based_fallback" | "none";

export interface ModelInfo {
  model_name: string;
  model_type: ModelType;
  artifact_available: boolean;
  trained_at?: string | null;
  feature_count?: number | null;
  dataset?: string | null;
  evaluation_available: boolean;
  evaluation_generated_at?: string | null;
}

export interface ModelInfoResponse {
  model_name: string;
  model_type: ModelType;
  artifact_available: boolean;
  trained_at: string | null;
  dataset: string | null;
  feature_count: number | null;
  features: string[] | null;
  train_skus: string[] | null;
  training_metrics: { mae?: number | null; rmse?: number | null };
  evaluation: {
    available: boolean;
    generated_at: string | null;
    summary: Record<string, number | null> | null;
  };
  hint: string | null;
}

export interface ExplanationBlock {
  classification_reason: string;
  method_reason: string;
  risk_reason: string;
  confidence_note: string;
}

export interface DecisionBlock {
  lead_time_days: number;
  lead_time_demand: number;
  safety_stock: number;
  safety_stock_method: string;
  reorder_point: number;
  service_level: number;
  inventory_gap: number;
  why: string;
  constraints?: {
    raw_order_quantity: number;
    final_order_quantity: number;
    moq?: number | null;
    order_multiple?: number | null;
    max_order_quantity?: number | null;
    policy_source?: "sku_policy" | "pattern_default" | string;
    constraints_applied: string[];
    moq_applied: boolean;
    order_multiple_applied: boolean;
    max_order_cap_applied: boolean;
    constrained: boolean;
    remaining_gap_after_order: number;
  };
  uncertainty?: {
    source: string;
    sigma: number;
    sample_count: number;
    lookback_days?: number | null;
    fallback_used: boolean;
    method?: string | null;
    horizon_days?: number | null;
    demand_pattern?: string | null;
  };
}

export interface SkuAnalysis {
  sku: string;
  risk: "HIGH" | "MEDIUM" | "LOW";
  risk_color: string;
  forecast: {
    p50: number;
    p90: number;
    daily: number[];
    full_horizon_daily?: number[];
    horizon_days?: number;
  };
  current_stock: number;
  recommended_order: number;
  action: string;
  demand_pattern: string;
  forecast_method: string;
  demand_source: DemandSource;
  // Optional for backward compatibility with older backend builds.
  forecast_source?: ForecastSource;
  decision?: DecisionBlock;
  model_info?: ModelInfo;
  explanation?: ExplanationBlock;
}

export interface RecentAnalysis {
  id: number;
  created_at: string;
  sku: string;
  risk: string | null;
  action: string | null;
  current_stock: number | null;
  recommended_order: number | null;
  demand_pattern: string | null;
  forecast_method: string | null;
  demand_source: string | null;
  forecast_source: string | null;
  model_type: string | null;
  model_name: string | null;
  artifact_available: boolean | null;
  lead_time_demand: number | null;
  safety_stock: number | null;
  reorder_point: number | null;
  inventory_gap: number | null;
  p50: number | null;
  p90: number | null;
}

export interface RecentAnalysesResponse {
  available: boolean;
  items: RecentAnalysis[];
  total: number;
  source?: string;
}

export interface StockLevel {
  sku: string;
  quantity_on_hand: number;
  quantity_reserved: number;
  quantity_available: number;
  source: string;
  recorded_at?: string | null;
}

export interface StockListResponse {
  items: StockLevel[];
  source: "database";
}

export interface KpiInterpretation {
  baseline: string;
  baseline_description: string;
  intelligent_description: string;
  assumptions: {
    lead_time_days: number;
    service_level: number;
    holding_cost_per_unit: number;
    stockout_cost_per_unit: number;
    simulation_window_days: number;
  };
  metric_meanings: Record<string, string>;
}

export interface KpiData {
  total_cost: number;
  fill_rate: number;
  cost_savings_pct: number;
  holding_cost: number;
  stockout_cost: number;
  naive_total_cost: number;
  intelligent_total_cost: number;
  skus_analyzed: number;
  interpretation?: KpiInterpretation;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
}

export type ModelMonitoringStatus =
  | "unavailable"
  | "insufficient_evidence"
  | "stable"
  | "warning"
  | "degraded";

export type ModelMonitoringBaselineProvenance =
  | "promotion_evidence"
  | "artifact_metadata"
  | "offline_backtest"
  | "unavailable"
  | string;

export interface ModelMonitoringSnapshot {
  model_artifact_id?: number | null;
  model_name: string;
  model_version?: string | null;
  lifecycle_status?: string | null;
  generated_at?: string | null;
  status: ModelMonitoringStatus;
  degradation_reason?: string | null;
  degradation_message?: string | null;
  evaluation_count: number;
  window_type?: string | null;
  window_size?: number | null;
  metric_wape?: number | null;
  metric_mae?: number | null;
  metric_rmse?: number | null;
  metric_bias?: number | null;
  metric_mase?: number | null;
  residual_mean?: number | null;
  residual_std?: number | null;
  baseline_wape?: number | null;
  baseline_provenance?: ModelMonitoringBaselineProvenance | null;
  wape_relative_change?: number | null;
  bias_ratio?: number | null;
  consecutive_degradation_count: number;
  created?: boolean | null;
}

export type AuthMode = "off" | "demo";

export interface AuthStatus {
  auth_mode: AuthMode;
  authenticated: boolean;
  user: string | null;
  login_required: boolean;
  note: string;
}

export interface DemandHistoryPoint {
  date: string;
  /** Legacy alias for ``units_sold`` — kept for backward compatibility. */
  demand: number;
  /** Real units sold on this calendar day. Not a forecast. */
  units_sold?: number;
}

export interface SkuHistorySummary {
  first_date: string;
  last_date: string;
  window_days_returned: number;
  total_units_sold: number;
  mean_units_per_day: number;
  peak_units_in_one_day: number;
  days_with_sales: number;
  days_with_zero_sales: number;
}

export interface SkuHistory {
  sku: string;
  available: boolean;
  history: DemandHistoryPoint[];
  /** Provenance: "recorded_history" when the values come from the dataset. */
  series_type?: "recorded_history";
  /** Human-readable: what the numeric values actually mean. */
  value_meaning?: "actual_units_sold";
  /** Where the numbers came from, usually "processed_dataset" for historical demand. */
  source?: "processed_dataset";
  description?: string;
  summary?: SkuHistorySummary | null;
  window_days_requested?: number;
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    // Credentials included so the demo-mode session cookie crosses origins
    // (frontend on :3000, backend on :8000). No effect when same-origin.
    const res = await fetch(`${this.baseUrl}${path}`, {
      credentials: "include",
      ...options,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const err = new Error(body.detail || `API error: ${res.status}`) as Error & { status?: number };
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  async getHealth(): Promise<HealthStatus> {
    return this.request("/health");
  }

  async getKpis(): Promise<KpiData> {
    return this.request("/api/kpis");
  }

  async getSkuDetails(): Promise<{ skus: SkuDetail[] }> {
    return this.request("/api/skus/details");
  }

  async analyzeSku(
    sku: string,
    currentStock: number,
    overrides?: { leadTimeDays?: number; serviceLevel?: number },
  ): Promise<SkuAnalysis> {
    const body: Record<string, unknown> = { sku, current_stock: currentStock };
    if (overrides?.leadTimeDays !== undefined) body.lead_time_days = overrides.leadTimeDays;
    if (overrides?.serviceLevel !== undefined) body.service_level = overrides.serviceLevel;
    return this.request("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  async getSkuHistory(sku: string, days: number = 30): Promise<SkuHistory> {
    return this.request(`/api/skus/${encodeURIComponent(sku)}/history?days=${days}`);
  }

  async getModelInfo(): Promise<ModelInfoResponse> {
    return this.request("/api/model-info");
  }

  async getModelMonitoring(): Promise<ModelMonitoringSnapshot> {
    return this.request(MODEL_MONITORING_ENDPOINT);
  }

  async getRecentAnalyses(limit: number = 10): Promise<RecentAnalysesResponse> {
    return this.request(`/api/analyses/recent?limit=${limit}`);
  }

  async getStockLevels(): Promise<StockListResponse> {
    return this.request("/api/stock");
  }

  async getStockForSku(sku: string): Promise<StockLevel> {
    return this.request(`/api/stock/${encodeURIComponent(sku)}`);
  }

  async updateStockForSku(
    sku: string,
    quantityOnHand: number,
    quantityReserved: number = 0,
  ): Promise<StockLevel> {
    return this.request(`/api/stock/${encodeURIComponent(sku)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quantity_on_hand: quantityOnHand,
        quantity_reserved: quantityReserved,
      }),
    });
  }

  async getAuthStatus(): Promise<AuthStatus> {
    return this.request("/api/auth/status");
  }

  async login(username: string, password: string): Promise<{ ok: boolean; user: string }> {
    return this.request("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
  }

  async logout(): Promise<{ ok: boolean }> {
    return this.request("/api/auth/logout", { method: "POST" });
  }

}

export const api = new ApiClient();

// Utility: deterministic demo stock levels
export function computeDemoStock(avgDemand: number, index: number): number {
  const multipliers = [0.3, 0.8, 2.0];
  return Math.max(1, Math.round(avgDemand * multipliers[index % 3]));
}
