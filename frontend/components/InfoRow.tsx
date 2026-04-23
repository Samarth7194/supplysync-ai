"use client";

import React from "react";

/**
 * Compact label → value row used across metadata panels.
 * Keeps the project's info surfaces visually consistent without a full table.
 */
export function InfoRow({
  label,
  value,
  hint,
  emphasis = "default",
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  emphasis?: "default" | "muted" | "warning";
}) {
  const valueColor =
    emphasis === "muted"
      ? "text-gray-400"
      : emphasis === "warning"
      ? "text-amber-300"
      : "text-white";

  return (
    <div className="flex items-baseline justify-between gap-3 text-xs" title={hint}>
      <span className="text-gray-500 uppercase tracking-wider">{label}</span>
      <span className={`${valueColor} font-medium text-right break-words`}>{value}</span>
    </div>
  );
}
