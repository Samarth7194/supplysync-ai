"use client";

import React, { useEffect, useState } from "react";
import { Lock, LogIn } from "lucide-react";
import { api, type AuthStatus } from "@/lib/api";

/**
 * Minimal client-side auth wrapper.
 *
 * Probes ``/api/auth/status`` on mount:
 *   - auth_mode=off            → render children immediately.
 *   - authenticated=true       → render children.
 *   - login_required=true      → render a small login card; on success,
 *                                refetch status and hand off to children.
 *
 * This is a demo-grade guard: a single shared credential, no user
 * management, no password hashing beyond constant-time compare. The UI
 * says so in one line so nobody mistakes it for an SSO portal.
 */
export function AuthGate({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [username, setUsername] = useState("demo");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  async function refreshStatus() {
    try {
      const s = await api.getAuthStatus();
      setStatus(s);
    } catch {
      // Treat a fetch failure as "no gate" — the individual pages already
      // render their own "backend unreachable" banner.
      setStatus({
        auth_mode: "off",
        authenticated: false,
        user: null,
        login_required: false,
        note: "Auth status unavailable (backend offline).",
      });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    // Canonical fetch-on-mount; setState is reached only after the await.
    // The strict react-hooks rule flags the transitive setter call.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshStatus();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setLoginError(null);
    try {
      await api.login(username, password);
      await refreshStatus();
    } catch (err) {
      setLoginError((err as Error).message || "Login failed");
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // No gate needed: children render as normal.
  if (!status?.login_required) {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center px-6">
      <div className="w-full max-w-sm bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-5">
        <div className="flex items-center gap-2">
          <Lock className="w-4 h-4 text-gray-400" />
          <h1 className="font-bold text-sm">Sign in to SupplySync</h1>
        </div>
        <p className="text-xs text-gray-500 leading-relaxed">
          This deployment runs in demo-auth mode. Use the shared credential
          configured via <code className="px-1 py-0.5 rounded bg-gray-800 text-gray-300">DEMO_USER</code>{" "}
          / <code className="px-1 py-0.5 rounded bg-gray-800 text-gray-300">DEMO_PASSWORD</code> on the backend.
          Not a production identity system.
        </p>
        <form onSubmit={handleSubmit} className="space-y-3">
          <label className="block">
            <span className="text-[11px] text-gray-500 uppercase tracking-wider">Username</span>
            <input
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="mt-1 w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </label>
          <label className="block">
            <span className="text-[11px] text-gray-500 uppercase tracking-wider">Password</span>
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </label>
          {loginError && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-md px-3 py-2">
              {loginError}
            </p>
          )}
          <button
            type="submit"
            disabled={submitting || !password}
            className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 transition-colors text-white text-sm font-medium py-2 rounded-md"
          >
            <LogIn className="w-4 h-4" />
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
