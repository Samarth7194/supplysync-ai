import type { Metadata } from "next";
import "./globals.css";
import { AuthGate } from "@/components/AuthGate";

export const metadata: Metadata = {
  title: "SupplySync AI - ML-Powered Inventory Optimization",
  description: "Intelligent inventory management using LightGBM demand forecasting, adaptive safety stock, and cost optimization across 4,900+ SKUs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased dark">
      <body className="min-h-full flex flex-col font-sans">
        <AuthGate>{children}</AuthGate>
      </body>
    </html>
  );
}
