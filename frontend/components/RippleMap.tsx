"use client";

import React, { useEffect, useRef } from "react";
import { Network } from "vis-network";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface RippleMapProps {
  data?: {
    nodes: any[];
    edges: any[];
  };
}

export function RippleMap({ data }: RippleMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current && data) {
      const options = {
        nodes: {
          shape: "dot",
          size: 30,
          font: { size: 14, color: "#ffffff" },
          borderWidth: 2,
          shadow: true,
        },
        edges: {
          width: 2,
          color: "#848484",
          arrows: { to: { enabled: true } },
        },
        physics: {
          enabled: true,
          solver: "forceAtlas2Based",
        },
      };

      const network = new Network(containerRef.current, data, options);
      return () => network.destroy();
    }
  }, [data]);

  return (
    <Card className="w-full h-[500px] bg-slate-900 border-slate-800 shadow-2xl overflow-hidden">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          🕸️ GNN Ripple Engine: Network Topology
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0 h-full">
        <div ref={containerRef} className="w-full h-full" />
      </CardContent>
    </Card>
  );
}
