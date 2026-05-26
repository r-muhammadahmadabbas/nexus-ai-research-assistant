"""
FastAPI application entry point.
Strictly handles app instantiation, middleware, router inclusion, and static file serving.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes import router as chat_router

app = FastAPI(
    title="Nexus AI — Research Assistant API",
    description="Multi-agent API for dynamic business research.",
    version="2.0.0",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router)

# ─── Serve Frontend ──────────────────────────────────────
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", include_in_schema=False)
async def root():
    """Serves the frontend single-page application."""
    return FileResponse(os.path.join(web_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
