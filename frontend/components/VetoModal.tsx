"use client";

import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

interface VetoModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApprove: () => void;
  onVeto: () => void;
  proposal?: {
    sku: string;
    cost: number;
    logic_chain: {
      sense: string;
      reason: string;
      plan: string;
    };
  };
}

export function VetoModal({ isOpen, onClose, onApprove, onVeto, proposal }: VetoModalProps) {
  if (!proposal) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[550px] bg-slate-950 border-slate-800 text-white shadow-3xl">
        <DialogHeader>
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="text-amber-500 w-6 h-6" />
            <DialogTitle className="text-xl">High-Cost Action Approval Required</DialogTitle>
          </div>
          <DialogDescription className="text-slate-400">
            A reorder decision exceeds the $10,000 threshold. Manual verification is mandatory.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <div className="flex justify-between items-center bg-slate-900/50 p-4 rounded-lg border border-slate-800">
            <div>
              <p className="text-sm text-slate-400">SKU Identifier</p>
              <h3 className="text-lg font-bold">{proposal.sku}</h3>
            </div>
            <div>
              <p className="text-sm text-slate-400 text-right">Total Expected Cost</p>
              <h3 className="text-lg font-bold text-amber-500">${proposal.cost.toLocaleString()}</h3>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-slate-500 uppercase tracking-widest">Logic Chain Trace</h4>
            <div className="space-y-3">
              <div className="flex gap-3">
                <Badge variant="outline" className="h-6 w-20 flex justify-center border-blue-500/50 text-blue-400">SENSE</Badge>
                <p className="text-sm text-slate-300">{proposal.logic_chain.sense}</p>
              </div>
              <div className="flex gap-3">
                <Badge variant="outline" className="h-6 w-20 flex justify-center border-purple-500/50 text-purple-400">REASON</Badge>
                <p className="text-sm text-slate-300">{proposal.logic_chain.reason}</p>
              </div>
              <div className="flex gap-3">
                <Badge variant="outline" className="h-6 w-20 flex justify-center border-green-500/50 text-green-400">PLAN</Badge>
                <p className="text-sm text-slate-300">{proposal.logic_chain.plan}</p>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="flex gap-2 sm:gap-0">
          <Button variant="destructive" onClick={onVeto} className="flex-1">
            VETO ACTION
          </Button>
          <Button variant="default" onClick={onApprove} className="flex-1 bg-green-600 hover:bg-green-700">
            <CheckCircle2 className="mr-2 h-4 w-4" /> COMMIT ACTION
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
