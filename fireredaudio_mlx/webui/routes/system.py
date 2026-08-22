"""System and hardware metrics API routes."""

from fastapi import APIRouter
from pydantic import BaseModel
from ..manager import model_manager
from ..config import QUALITY_PRESETS

router = APIRouter(prefix="/system", tags=["System"])


class LoadModelRequest(BaseModel):
    model_path: str


@router.get("/status")
@router.get("/model/status")
def get_system_status():
    return model_manager.get_status()


@router.get("/presets")
def get_presets():
    return QUALITY_PRESETS


@router.post("/model/load")
def trigger_model_load(req: LoadModelRequest):
    model_manager.reload_model(req.model_path)
    return {"message": "模型后台加载已触发", "status": model_manager.get_status()}


@router.post("/cache/clear")
def clear_metal_cache():
    model_manager.clear_cache()
    return {"message": "Metal 显存缓存已清理", "memory": model_manager.get_memory_info()}
