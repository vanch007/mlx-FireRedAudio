"""API Routes for FireRedAudio WebUI."""

from fastapi import APIRouter
from fastapi import HTTPException
from ..jobs import job_queue
from ..manager import model_manager
from .system import router as system_router
from .jobs import router as jobs_router
from .projects import router as projects_router
from .assets import router as assets_router
from .voices import router as voices_router
from .media import router as media_router
from .sse import router as sse_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system_router)
api_router.include_router(jobs_router)
api_router.include_router(projects_router)
api_router.include_router(assets_router)
api_router.include_router(voices_router)
api_router.include_router(media_router)
api_router.include_router(sse_router)


@api_router.get("/model/status")
def get_model_status_alias():
    return model_manager.get_status()


@api_router.get("/results/{result_id}")
def get_result(result_id: str):
    job = job_queue.get_job(result_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到指定的结果或任务")
    return {
        "id": job.id,
        "task": job.task,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "result": job.result,
        "error": job.error,
        "latency_seconds": job.latency_seconds,
        "project_id": job.project_id,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }
