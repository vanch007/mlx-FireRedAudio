"""FastAPI Application factory for FireRedAudio Studio."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIST_DIR, DEFAULT_WORKSPACE_DIR, DEFAULT_MODEL_DIR
from .jobs import job_queue
from .manager import model_manager
from .workspace import workspace_store
from .routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Start persistent JobQueue worker
    await job_queue.start()
    # 2. Trigger asynchronous background model pre-loading
    model_manager.start_background_load()
    yield
    # Shutdown cleanup
    model_manager.clear_cache()


def create_app(model_path: str = None, workspace_dir: str = None) -> FastAPI:
    if model_path:
        model_manager.model_path = model_path
    if workspace_dir:
        workspace_store.reconfigure(workspace_dir)
        job_queue.jobs_dir = workspace_store.jobs_dir

    app = FastAPI(
        title="FireRedAudio Studio API",
        description="Local-First MLX Audio Language Model WebUI & Workflow Studio",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Mount compiled React SPA if built
    if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="frontend-assets")

        @app.get("/", response_class=HTMLResponse)
        def serve_index():
            return FileResponse(str(FRONTEND_DIST_DIR / "index.html"))

        @app.exception_handler(404)
        async def spa_404_handler(request, exc):
            if request.url.path.startswith("/api/"):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
            return FileResponse(str(FRONTEND_DIST_DIR / "index.html"))
    else:
        @app.get("/", response_class=HTMLResponse)
        def fallback_index():
            return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>FireRedAudio Studio</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 2.5rem; border-radius: 1rem; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); text-align: center; max-width: 500px; border: 1px solid #334155; }
        h1 { font-size: 1.75rem; margin-bottom: 0.5rem; color: #38bdf8; }
        p { color: #94a3b8; line-height: 1.5; margin-bottom: 1.5rem; }
        a { display: inline-block; background: #0284c7; color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none; font-weight: 500; }
        a:hover { background: #0369a1; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🎙️ FireRedAudio Studio API 就绪</h1>
        <p>后端 FastAPI + MLX 推理服务已成功启动。正在构建 React 前端界面，您也可以直接访问交互式 API 文档。</p>
        <a href="/docs" target="_blank">查看 API 文档 (/docs)</a>
    </div>
</body>
</html>"""

    return app
