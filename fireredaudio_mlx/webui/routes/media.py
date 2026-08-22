"""Secure media streaming API routes."""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..workspace import workspace_store

router = APIRouter(prefix="/media", tags=["Media"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _is_safe_audio_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
        root = workspace_store.root_dir.resolve()
        if not str(resolved).startswith(str(root)):
            return False
        return resolved.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS and resolved.exists()
    except Exception:
        return False


@router.get("/assets/{filename}")
def stream_asset(filename: str):
    safe_name = Path(filename).name
    file_path = workspace_store.assets_dir / safe_name
    if not _is_safe_audio_path(file_path):
        raise HTTPException(status_code=404, detail="找不到请求的音频文件或文件类型不受支持")
    return FileResponse(path=str(file_path), media_type="audio/wav")


@router.get("/outputs/{filename}")
def stream_output(filename: str):
    safe_name = Path(filename).name
    file_path = workspace_store.outputs_dir / safe_name
    if not _is_safe_audio_path(file_path):
        raise HTTPException(status_code=404, detail="找不到请求的输出音频文件或文件类型不受支持")
    return FileResponse(path=str(file_path), media_type="audio/wav")


@router.get("/{asset_id}")
def stream_asset_by_id(asset_id: str):
    asset = workspace_store.get_asset(asset_id)
    if asset:
        file_path = workspace_store.get_asset_file_path(asset)
        if _is_safe_audio_path(file_path):
            return FileResponse(path=str(file_path), media_type="audio/wav")
    raise HTTPException(status_code=404, detail="找不到指定的音频素材或音频文件不存在")
