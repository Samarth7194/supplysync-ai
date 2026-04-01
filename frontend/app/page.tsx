"use client";

import React, { useState, useEffect } from "react";
import { 
  Activity, 
  Terminal, 
  ShieldAlert, 
  Settings, 
  Network as NetworkIcon, 
  LineChart as LineChartIcon,
  MessageSquare,
  AlertTriangle,
  Cpu
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { RippleMap } from "@/components/RippleMap";
import { InventoryFanChart } from "@/components/InventoryFanChart";
import { VetoModal } from "@/components/VetoModal";

const MOCK_TOPOLOGY = {
  nodes: [
    { id: 1, label: "Supplier: ChipCorp", color: "#ef4444", font: { color: "white" } },
    { id: 2, label: "Hub: Berlin", color: "#eab308" },
    { id: 3, label: "Warehouse: LA", color: "#22c55e" },
    { id: 4, label: "SKU: GPU-A100", color: "#3b82f6" },
  ],
  edges: [
    { from: 1, to: 2, label: "Risk Ripple", color: "#ef4444" },
    { from: 2, to: 3 },
    { from: 3, to: 4 },
  ],
};

export default function MissionControl() {
  const [logs, setLogs] = useState<{ phase: string; message: string; type?: string }[]>([]);
  const [isVetoOpen, setIsVetoOpen] = useState(false);
  const [topology, setTopology] = useState(MOCK_TOPOLOGY);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    // 1. Fetch Topology
    fetch(`${apiUrl}/api/network-topology`)
      .then(res => res.json())
      .then(data => setTopology(data))
      .catch(err => console.error("Topology fetch error:", err));

    // 2. Real-time CoT Reasoning Stream
    const eventSource = new EventSource(`${apiUrl}/api/stream`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const phaseColors: Record<string, string> = {
        "SENSE": "#3b82f6", // blue
        "REASON": "#a855f7", // purple
        "PLAN": "#22c55e", // green
        "ACT": "#f59e0b", // amber
        "ALERT": "#ef4444" // red
      };

      setLogs((prev) => [...prev, { 
        phase: data.phase, 
        message: data.message, 
        type: phaseColors[data.phase] || "#94a3b8" 
      }]);

      if (data.phase === "ALERT") setIsVetoOpen(true);
    };

    eventSource.onerror = (err) => {
      console.error("SSE Error:", err);
      eventSource.close();
    };

    return () => eventSource.close();
  }, []);

  return (
    <div className="min-h-screen bg-black text-slate-100 font-sans selection:bg-blue-500/30">
      {/* Top Header */}
      <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Cpu className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">SupplySync AI</h1>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
              Sovereign Node: uri:agent:supplysync-01
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <Badge variant="outline" className="border-slate-700 text-slate-400 bg-slate-800/50 hidden md:flex">
            BFT Consensus: Active
          </Badge>
          <Badge variant="outline" className="border-slate-700 text-slate-400 bg-slate-800/50 hidden md:flex">
            SAA Scenarios: 1,000
          </Badge>
          <Separator orientation="vertical" className="h-6 bg-slate-800 mx-1" />
          <Button variant="ghost" size="icon" className="text-slate-400 hover:text-white">
            <Settings className="w-5 h-5" />
          </Button>
        </div>
      </header>

      {/* Main Layout */}
      <main className="p-6 grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-4rem)]">
        {/* Left Sidebar: A2A Negotiation Feed */}
        <aside className="lg:col-span-1 border border-slate-800 bg-slate-900/30 rounded-2xl overflow-hidden flex flex-col shadow-inner">
          <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-blue-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider">A2A Negotiation Feed</h2>
            </div>
            <Badge variant="secondary" className="bg-blue-500/10 text-blue-400 border-none text-[10px]">LIVE</Badge>
          </div>
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              <NegotiationItem 
                agent="Supplier: ChipCorp" 
                status="COUNTER-OFFER" 
                time="2m ago" 
                detail="Partial qty (150) proposed due to capacity."
              />
              <NegotiationItem 
                agent="Hub: Berlin" 
                status="ACKNOWLEDGED" 
                time="5m ago" 
                detail="Route valid_at 2026-03-24 confirmed."
              />
              <NegotiationItem 
                agent="Warehouse: LA" 
                status="SETTLED" 
                time="15m ago" 
                detail="Settlement signature verified (ED25519)."
              />
            </div>
          </ScrollArea>
        </aside>

        {/* Center Canvas: Charts & GNN Map */}
        <section className="lg:col-span-3 space-y-6 flex flex-col h-full">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 flex-1">
            <RippleMap data={topology} />
            <InventoryFanChart />
          </div>

          {/* Bottom Console: Reasoning CoT */}
          <Card className="h-48 bg-black border-slate-800 shadow-2xl overflow-hidden flex flex-col">
            <CardHeader className="py-3 px-4 bg-slate-900/50 border-b border-slate-800">
              <CardTitle className="text-xs uppercase tracking-[0.2em] font-bold text-slate-500 flex items-center gap-2">
                <Terminal className="w-4 h-4" /> Reasoning Chain-of-Thought
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1 bg-[#050505]">
              <ScrollArea className="h-full font-mono text-[13px] p-4 text-slate-400">
                {logs.map((log, idx) => (
                  <div key={idx} className="mb-1 flex gap-4">
                    <span className="text-slate-600">[{new Date().toLocaleTimeString()}]</span>
                    <span className="font-bold w-16" style={{ color: log.type }}>{log.phase}:</span>
                    <span style={{ color: log.type }}>{log.message}</span>
                  </div>
                ))}
              </ScrollArea>
            </CardContent>
          </Card>
        </section>
      </main>

      <VetoModal 
        isOpen={isVetoOpen}
        onClose={() => setIsVetoOpen(false)}
        onApprove={() => setIsVetoOpen(false)}
        onVeto={() => setIsVetoOpen(false)}
        proposal={{
          sku: "GPU-A100",
          cost: 18500,
          logic_chain: {
            sense: "Detected critical stock variance (12/35) in Warehouse: LA.",
            reason: "GNN Ripple Engine identifies 78% probability of downstream Stockout-Cascades.",
            plan: "Benders-Optimized SMILP recommends 200 unit reorder to minimize E[InventoryCost]."
          }
        }}
      />
    </div>
  );
}

function NegotiationItem({ agent, status, time, detail }: any) {
  return (
    <div className="p-3 bg-slate-800/20 border border-slate-800/50 rounded-xl hover:bg-slate-800/40 transition-colors">
      <div className="flex justify-between items-start mb-1">
        <p className="text-[11px] font-bold text-slate-300">{agent}</p>
        <span className="text-[9px] text-slate-500">{time}</span>
      </div>
      <Badge className="bg-slate-700 text-slate-400 border-none text-[9px] mb-2">{status}</Badge>
      <p className="text-[10px] text-slate-400 leading-relaxed">{detail}</p>
    </div>
  );
}
