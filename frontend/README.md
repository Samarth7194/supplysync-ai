# SupplySync Frontend

Next.js 16 App Router dashboard for the SupplySync AI backend. It renders the KPI overview, SKU table, per-SKU detail page, stock editing, and provenance-labeled forecasts.

The root [README.md](../README.md) has the full project context. This file documents the frontend package only.

## Prerequisites

- Node.js 20 or newer
- Backend running at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`)

## Install And Run

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Scripts

| Script | What it does |
|---|---|
| `npm run dev` | Next dev server with hot reload |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint with `eslint-config-next` |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | Typecheck + lint |
| `npm run clean` | Remove local build artifacts |

## Structure

```text
frontend/
|-- app/
|   |-- layout.tsx
|   |-- page.tsx
|   |-- sku/[id]/page.tsx
|   `-- globals.css
|-- components/
|   `-- DataSourceBadge.tsx
|-- hooks/
|   `-- use-mobile.ts
|-- lib/
|   |-- api.ts
|   |-- env.ts
|   |-- stock.ts
|   `-- utils.ts
|-- public/
|-- next.config.ts
|-- tsconfig.json
`-- package.json
```

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL the browser uses to reach the backend |

To override locally, copy `.env.example` to `.env.local` and edit the value.
