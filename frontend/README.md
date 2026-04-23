# SupplySync Frontend

Next.js 16 (App Router) dashboard for the SupplySync AI backend. Renders the KPI overview, SKU table, and a per-SKU detail page with real historical demand and provenance-labeled forecasts.

The root [README.md](../README.md) has the full project context (architecture, forecasting logic, provenance vocabulary). This file documents the frontend package only.

## Prerequisites

- Node.js **≥ 20**
- A running backend at the URL in `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). See the root README for backend setup.

## Install & Run

```bash
npm install
npm run dev          # http://localhost:3000
```

## Scripts

| Script | What it does |
|---|---|
| `npm run dev` | Next dev server with hot reload |
| `npm run build` | Production build (Next standalone output) |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint with `eslint-config-next` |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | Typecheck + lint (used by CI) |
| `npm run clean` | Remove `.next/`, `out/`, and `tsconfig.tsbuildinfo` |

## Structure

```
frontend/
├── app/
│   ├── layout.tsx                # Root layout, fonts, metadata
│   ├── page.tsx                  # Dashboard (/)
│   ├── sku/[id]/page.tsx         # SKU detail (/sku/<id>)
│   ├── api/copilotkit/route.ts   # Optional CopilotKit endpoint (scaffolding)
│   └── globals.css
├── components/
│   ├── DataSourceBadge.tsx       # Shared provenance pill
│   └── ui/                       # shadcn/ui primitives
├── hooks/
│   └── use-mobile.ts
├── lib/
│   ├── api.ts                    # Typed client + shared types
│   └── utils.ts                  # cn() and small utils
├── public/                       # Static assets
├── eslint.config.mjs
├── next.config.ts                # output: "standalone" for Docker image
├── postcss.config.mjs
├── components.json               # shadcn/ui config
├── tsconfig.json
└── package.json
```

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL the browser uses to reach the backend |

To override, copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

The value is read at both runtime and build time through `lib/env.ts`.

## Notes

- The `app/api/copilotkit/route.ts` endpoint is scaffolding for a future chat-style agent. It compiles and serves, but nothing in the dashboard currently calls it. Removing it and the `@copilotkit/*` deps is safe if you don't plan to wire an agent.
- UI primitives in `components/ui/` are generated via shadcn/ui (`npx shadcn@latest add <component>`). `shadcn` is a devDependency so you can re-run the generator without a global install.
