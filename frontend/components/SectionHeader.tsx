"use client";

import React from "react";
import { cn } from "@/lib/utils";

/**
 * Unified section header used across the dashboard and SKU-detail page.
 *
 * Keeping a single component means every section on the app has the same
 * title weight, subtitle tone, and spacing — which is where "feels like an
 * intentional product" actually comes from.
 */
export function SectionHeader({
  eyebrow,
  title,
  subtitle,
  right,
  className = "",
  compact = false,
}: {
  eyebrow?: React.ReactNode;
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
  /** Tighter vertical rhythm when used inside an already-padded card. */
  compact?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-4 flex-wrap",
        compact ? "mb-3" : "mb-5",
        className,
      )}
    >
      <div>
        {eyebrow && (
          <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.12em] mb-1.5">
            {eyebrow}
          </p>
        )}
        <h2 className={cn("font-semibold text-white", compact ? "text-sm" : "text-base")}>
          {title}
        </h2>
        {subtitle && (
          <p className="text-xs text-gray-500 mt-1 max-w-3xl leading-relaxed">
            {subtitle}
          </p>
        )}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}
