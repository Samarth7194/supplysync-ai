"use client";

import React from "react";
import { AlertTriangle, Info } from "lucide-react";

/**
 * Section-level empty/unavailable state.
 *
 * Kept intentionally small — one icon + title + optional hint + optional
 * action — so every "data not here" message looks the same across the app.
 */
export function EmptyState({
  title,
  hint,
  tone = "neutral",
  icon: Icon,
  className = "",
  children,
}: {
  title: string;
  hint?: React.ReactNode;
  tone?: "neutral" | "warning";
  icon?: React.ComponentType<{ className?: string }>;
  className?: string;
  children?: React.ReactNode;
}) {
  const container =
    tone === "warning"
      ? "border-amber-500/30 bg-amber-500/5"
      : "border-gray-800 bg-transparent";
  const titleColor = tone === "warning" ? "text-amber-200" : "text-gray-300";
  const IconComp = Icon ?? (tone === "warning" ? AlertTriangle : Info);

  return (
    <div
      className={`flex flex-col items-center justify-center text-center px-6 py-8 border border-dashed rounded-lg ${container} ${className}`}
    >
      <IconComp
        className={`w-4 h-4 mb-2 ${tone === "warning" ? "text-amber-300" : "text-gray-500"}`}
      />
      <p className={`text-sm font-semibold ${titleColor}`}>{title}</p>
      {hint && (
        <p className="text-xs text-gray-500 mt-1 max-w-md leading-relaxed">{hint}</p>
      )}
      {children && <div className="mt-3">{children}</div>}
    </div>
  );
}
