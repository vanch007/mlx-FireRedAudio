"""Asset management and audio upload API routes."""

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..workspace import workspace_store, Asset

router = APIRouter(prefix="/assets", tags=["Assets"])


@router.post("/upload", response_model=Asset)
async def upload_asset(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    source: str = Form("upload"),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传的文件内容为空")
    try:
        return workspace_store.save_uploaded_asset(
            filename=file.filename or "audio.wav",
            content=content,
            project_id=project_id,
            source=source,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[Asset])
def list_assets(project_id: Optional[str] = None):
    return workspace_store.list_assets(project_id=project_id)


@router.get("/{asset_id}", response_model=Asset)
def get_asset(asset_id: str):
    a = workspace_store.get_asset(asset_id)
    if not a:
        raise HTTPException(status_code=404, detail="找不到指定的音频素材")
    return a


@router.delete("/{asset_id}")
def delete_asset(asset_id: str):
    try:
        ok = workspace_store.delete_asset(asset_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="找不到指定的音频素材")
    return {"message": "素材已删除", "asset_id": asset_id}
