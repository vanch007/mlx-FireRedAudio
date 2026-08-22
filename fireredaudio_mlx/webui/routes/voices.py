"""Voice profile management API routes."""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..workspace import workspace_store, VoiceProfile

router = APIRouter(prefix="/voices", tags=["Voices"])


class CreateVoiceRequest(BaseModel):
    name: str
    prompt_text: str
    audio_asset_id: str
    language: str = "zh"
    description: str = ""


@router.get("", response_model=List[VoiceProfile])
def list_voices():
    return workspace_store.list_voices()


@router.post("", response_model=VoiceProfile)
def create_voice(req: CreateVoiceRequest):
    asset = workspace_store.get_asset(req.audio_asset_id)
    if not asset:
        raise HTTPException(status_code=400, detail=f"关联的参考音频素材不存在: {req.audio_asset_id}")
    return workspace_store.save_voice(
        name=req.name,
        prompt_text=req.prompt_text,
        audio_asset_id=req.audio_asset_id,
        language=req.language,
        description=req.description,
    )


@router.get("/{voice_id}", response_model=VoiceProfile)
def get_voice(voice_id: str):
    v = workspace_store.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="找不到指定的声音配置")
    return v


@router.delete("/{voice_id}")
def delete_voice(voice_id: str):
    ok = workspace_store.delete_voice(voice_id)
    if not ok:
        raise HTTPException(status_code=404, detail="找不到指定的声音配置")
    return {"message": "声音配置已删除", "voice_id": voice_id}
