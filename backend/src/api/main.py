# src/api/main.py

from fastapi import FastAPI
from api.routes import forecast, reorder, kpis

app = FastAPI(title="SupplySync AI")

app.include_router(forecast.router)
app.include_router(reorder.router)
app.include_router(kpis.router)
