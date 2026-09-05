# SupplySync AI — developer entrypoint.
#
# Keeps the common commands discoverable without forcing a specific language
# or build tool. All targets are safe to re-run.

.PHONY: help install bootstrap test backend frontend dev demo clean

help:
	@echo "SupplySync AI — common targets"
	@echo "  make install      Install Python + Node dependencies"
	@echo "  make bootstrap    Generate parquet, train model, compute KPIs"
	@echo "  make test         Run the backend test suite"
	@echo "  make backend      Run the FastAPI backend on :8000 (reload)"
	@echo "  make frontend     Run the Next.js dashboard on :3000"
	@echo "  make demo         End-to-end demo bring-up (install + bootstrap)"
	@echo "  make clean        Drop generated parquet / sqlite / next build"

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

bootstrap:
	cd backend && python scripts/bootstrap.py

test:
	cd backend && python -m pytest tests/ -q

backend:
	cd backend && uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

demo: install bootstrap
	@echo ""
	@echo "Demo setup complete. Open two shells:"
	@echo "  1) make backend"
	@echo "  2) make frontend"
	@echo "Then visit http://localhost:3000"

clean:
	rm -f data/processed/*.parquet
	rm -f backend/data/supplysync.db*
	rm -rf frontend/.next frontend/out
