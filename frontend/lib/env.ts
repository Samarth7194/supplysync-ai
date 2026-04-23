/**
 * Centralized access to public environment variables.
 *
 * Keeping a single entry point means every page/hook imports from here
 * instead of re-reading `process.env.NEXT_PUBLIC_*` inline, which:
 *   - gives TypeScript a place to narrow the types, and
 *   - makes it obvious which env vars the browser actually needs.
 *
 * Only `NEXT_PUBLIC_*` values are safe to read here; anything else would be
 * inlined at build time and could leak secrets to the browser bundle.
 */

function readUrl(value: string | undefined, fallback: string): string {
  if (!value) return fallback;
  const trimmed = value.trim().replace(/\/+$/, "");
  return trimmed || fallback;
}

export const env = {
  /** Backend base URL. Defaults to localhost for dev. */
  apiUrl: readUrl(process.env.NEXT_PUBLIC_API_URL, "http://localhost:8000"),
} as const;

export type Env = typeof env;
