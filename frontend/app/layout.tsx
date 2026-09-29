import type { Metadata } from "next";
import "./globals.css";
import { AuthGate } from "@/components/AuthGate";

export const metadata: Metadata = {
  title: "SupplySync AI - ML-Powered Inventory Optimization",
  description: "Intelligent inventory management with evidence-routed demand forecasting and adaptive safety stock — a 20-SKU demo of a 4,900-SKU dataset.",
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
