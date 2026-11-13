"""Main Application Entry Point for Local Dataset Factory."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from .api.routes import router as api_router

app = FastAPI(
    title="Local Hybrid Dataset Factory",
    description="100% Local Document-to-Fine-Tuning Dataset Factory with Intelligent Routing and Evidence Verification",
    version="1.0.0"
)

# Allow local frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

FRONTEND_INDEX = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")


@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return "<h1>Local Dataset Factory API Ready. Visit /docs for OpenAPI specifications.</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ForgeDataset.main:app", host="127.0.0.1", port=8000, reload=True)
