from fastapi import FastAPI
from api.endpoints import router as api_router

app = FastAPI(title="Data Processing for OSAA")
app.include_router(api_router)
