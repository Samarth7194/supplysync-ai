"use client";

import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface FanChartProps {
  data?: any[];
}

export function InventoryFanChart({ data }: FanChartProps) {
  // Use data prop or fallback to realistic dummy data
  const defaultData = data || [
    { name: "Week 1", p10: 80, p50: 100, p90: 130 },
    { name: "Week 2", p10: 85, p50: 110, p90: 150 },
    { name: "Week 3", p10: 90, p50: 125, p90: 180 },
    { name: "Week 4", p10: 110, p50: 145, p90: 220 },
    { name: "Week 5", p10: 120, p50: 170, p90: 280 },
  ];

  return (
    <Card className="w-full bg-slate-900 border-slate-800 shadow-2xl">
      <CardHeader>
        <CardTitle className="text-white">📈 TFT Probabilistic Forecast (Fan Chart)</CardTitle>
      </CardHeader>
      <CardContent className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={defaultData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorP90" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155", color: "#f8fafc" }}
            />
            <Legend />
            {/* The "Fan" area between P10 and P90 */}
            <Area
              type="monotone"
              dataKey="p90"
              stroke="#ef4444"
              fillOpacity={1}
              fill="url(#colorP90)"
              name="90% Upper Bound"
            />
            <Area
              type="monotone"
              dataKey="p10"
              stroke="#22c55e"
              fill="#1e293b"
              fillOpacity={1}
              name="10% Lower Bound"
            />
            {/* The median line */}
            <Area
              type="monotone"
              dataKey="p50"
              stroke="#3b82f6"
              fill="transparent"
              strokeWidth={3}
              name="Median Forecast"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
