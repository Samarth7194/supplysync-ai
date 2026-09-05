# Resume Project Entry

Ready-to-paste project entry for a resume, targeting AI/ML Engineer internship and entry-level roles.

## A. Recommended Project Title

**SupplySync AI — ML Inventory Decision Support with Controlled MLOps**

## B. One-Line Tech Stack

Python · FastAPI · LightGBM · SQLAlchemy · PostgreSQL · Next.js · TypeScript

## C. Resume Bullets (full version — 3 bullets)

- Built a hybrid demand-forecasting pipeline (LightGBM, Croston-SBA, conservative buffer) that routes ~4,900 SKUs by demand pattern, converting forecasts into uncertainty-aware, constrained reorder recommendations.
- Designed a controlled MLOps lifecycle — forecast evaluation, performance monitoring, retraining recommendations, candidate training/evaluation, and human-approved promotion/rollback with a full audit trail.
- Deployed a full-stack app (Next.js/Vercel, FastAPI/Render, PostgreSQL/Neon) with 280+ backend tests, CI-validated migrations, and a historical-replay mechanism that demonstrates monitoring without fabricating live data.

## 2-Bullet Compact Version (for a one-page resume)

- Built an ML inventory decision-support system with hybrid demand forecasting (LightGBM/Croston-SBA) and uncertainty-aware safety stock, routing ~4,900 SKUs to the appropriate method per demand pattern.
- Implemented a controlled MLOps lifecycle — monitoring, degradation detection, candidate evaluation, and human-approved model promotion/rollback — deployed full-stack with 280+ tests in CI.

## Notes on Framing

- Do not describe this as autonomous, enterprise-grade, or real-time AI — it is a decision-support prototype on historical retail data.
- If asked to shorten further, keep the MLOps lifecycle bullet — it is the most differentiated part relative to typical forecasting-only portfolio projects.
- Pair with the [Interview Guide](./INTERVIEW_GUIDE.md) before discussing this project live.
