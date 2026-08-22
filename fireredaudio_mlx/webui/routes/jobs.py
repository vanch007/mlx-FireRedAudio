"""Job management API routes."""

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from ..jobs import job_queue, JobCreate, Job

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("", response_model=Job)
async def create_job(job_create: JobCreate):
    return await job_queue.submit_job(job_create)


@router.get("", response_model=List[Job])
def list_jobs(project_id: Optional[str] = None, limit: int = 50):
    return job_queue.list_jobs(project_id=project_id, limit=limit)


@router.get("/{job_id}", response_model=Job)
def get_job_detail(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到指定的任务")
    return job


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    ok = await job_queue.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="无法取消该任务（任务可能已完成、已失败或正在运行中）")
    return {"message": "任务已成功取消", "job_id": job_id}


@router.post("/{job_id}/retry", response_model=Job)
async def retry_job(job_id: str):
    try:
        return await job_queue.retry_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
