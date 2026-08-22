"""Workspace storage manager for projects, assets, voices, and jobs."""

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import soundfile as sf
from pydantic import BaseModel, Field

from .config import DEFAULT_WORKSPACE_DIR


def atomic_write_json(file_path: Path, data: Union[BaseModel, Dict[str, Any]]):
    """Atomically write JSON content to disk via temporary file rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
    content = data.model_dump_json(indent=2) if isinstance(data, BaseModel) else json.dumps(data, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        tmp_path.replace(file_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class Asset(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    file_path: str
    media_url: str = ""
    file_size: int = 0
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    source: str = "upload"  # upload | generation | reference
    project_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class VoiceProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    prompt_text: str
    audio_asset_id: str
    language: str = "zh"
    description: str = ""
    created_at: float = Field(default_factory=time.time)


class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    asset_ids: List[str] = Field(default_factory=list)
    job_ids: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class WorkspaceStore:
    def __init__(self, root_dir: Optional[Path] = None):
        self.reconfigure(root_dir or DEFAULT_WORKSPACE_DIR)

    def reconfigure(self, root_dir: Union[str, Path]):
        """Reconfigure workspace root directory dynamically (e.g. for isolated test suites)."""
        self.root_dir = Path(root_dir or DEFAULT_WORKSPACE_DIR).resolve()
        self.projects_dir = self.root_dir / "projects"
        self.assets_dir = self.root_dir / "assets"
        self.voices_dir = self.root_dir / "voices"
        self.jobs_dir = self.root_dir / "jobs"
        self.outputs_dir = self.root_dir / "outputs"

        for d in [self.projects_dir, self.assets_dir, self.voices_dir, self.jobs_dir, self.outputs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_asset_file_path(self, asset: Asset) -> Path:
        """Resolve asset file path against root directory if relative, or return absolute path."""
        p = Path(asset.file_path)
        if not p.is_absolute():
            return (self.root_dir / p).resolve()
        return p.resolve()

    # --- Asset Management ---
    def probe_audio(self, file_path: Path) -> Dict[str, Any]:
        try:
            info = sf.info(str(file_path))
            return {
                "duration": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "file_size": file_path.stat().st_size,
            }
        except Exception:
            try:
                cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration:stream=sample_rate,channels",
                    "-of", "json", str(file_path)
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                probe = json.loads(out)
                dur = float(probe.get("format", {}).get("duration", 0.0))
                streams = probe.get("streams", [{}])
                sr = int(streams[0].get("sample_rate", 16000)) if streams else 16000
                ch = int(streams[0].get("channels", 1)) if streams else 1
                return {
                    "duration": round(dur, 2),
                    "sample_rate": sr,
                    "channels": ch,
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                }
            except Exception:
                pass
            return {
                "duration": 0.0,
                "sample_rate": 0,
                "channels": 1,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
            }

    def save_uploaded_asset(self, filename: str, content: bytes, project_id: Optional[str] = None, source: str = "upload") -> Asset:
        asset_id = str(uuid.uuid4())[:8]
        clean_name = Path(filename).name
        dest_filename = f"{asset_id}_{filename}"
        dest_path = self.assets_dir / dest_filename
        with open(dest_path, "wb") as f:
            f.write(content)

        meta = self.probe_audio(dest_path)
        if meta["duration"] <= 0 or meta["sample_rate"] <= 0:
            dest_path.unlink(missing_ok=True)
            raise ValueError(f"上传的文件 '{clean_name}' 无法解析为有效音频，或时长为0。")

        asset = Asset(
            id=asset_id,
            name=filename,
            file_path=str(dest_path),
            media_url=f"/api/v1/media/assets/{dest_filename}",
            file_size=meta["file_size"],
            duration=meta["duration"],
            sample_rate=meta["sample_rate"],
            channels=meta["channels"],
            source=source,
            project_id=project_id,
        )

        # Save asset meta JSON
        atomic_write_json(self.assets_dir / f"{asset_id}.json", asset)

        if project_id:
            self.add_asset_to_project(project_id, asset_id)

        return asset

    def list_assets(self, project_id: Optional[str] = None) -> List[Asset]:
        assets = []
        for p in self.assets_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    asset = Asset(**data)
                    if project_id is None or asset.project_id == project_id:
                        assets.append(asset)
            except Exception:
                continue
        assets.sort(key=lambda a: a.created_at, reverse=True)
        return assets

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        meta_file = self.assets_dir / f"{asset_id}.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return Asset(**json.load(f))
        except Exception:
            return None

    def delete_asset(self, asset_id: str) -> bool:
        asset = self.get_asset(asset_id)
        if not asset:
            return False

        # Check if referenced by any VoiceProfile
        for v in self.list_voices():
            if v.audio_asset_id == asset_id:
                raise ValueError(f"无法删除素材：该素材已被声音模板 '{v.name}' (#{v.id}) 引用。")

        try:
            Path(asset.file_path).unlink(missing_ok=True)
            (self.assets_dir / f"{asset_id}.json").unlink(missing_ok=True)
            # Remove reference from projects
            for proj in self.list_projects():
                if asset_id in proj.asset_ids:
                    proj.asset_ids = [aid for aid in proj.asset_ids if aid != asset_id]
                    proj.updated_at = time.time()
                    atomic_write_json(self.projects_dir / f"{proj.id}.json", proj)
            return True
        except Exception:
            return False

    # --- Voice Management ---
    def save_voice(self, name: str, prompt_text: str, audio_asset_id: str, language: str = "zh", description: str = "") -> VoiceProfile:
        voice_id = str(uuid.uuid4())[:8]
        voice = VoiceProfile(
            id=voice_id,
            name=name,
            prompt_text=prompt_text,
            audio_asset_id=audio_asset_id,
            language=language,
            description=description,
        )
        atomic_write_json(self.voices_dir / f"{voice_id}.json", voice)
        return voice

    def list_voices(self) -> List[VoiceProfile]:
        voices = []
        for p in self.voices_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    voices.append(VoiceProfile(**json.load(f)))
            except Exception:
                continue
        voices.sort(key=lambda v: v.created_at, reverse=True)
        return voices

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        meta_file = self.voices_dir / f"{voice_id}.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return VoiceProfile(**json.load(f))
        except Exception:
            return None

    def delete_voice(self, voice_id: str) -> bool:
        meta_file = self.voices_dir / f"{voice_id}.json"
        if meta_file.exists():
            meta_file.unlink()
            return True
        return False

    # --- Project Management ---
    def create_project(self, name: str, description: str = "") -> Project:
        project_id = str(uuid.uuid4())[:8]
        project = Project(id=project_id, name=name, description=description)
        atomic_write_json(self.projects_dir / f"{project_id}.json", project)
        return project

    def update_project(self, project_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Optional[Project]:
        proj = self.get_project(project_id)
        if not proj:
            return None
        if name is not None:
            proj.name = name
        if description is not None:
            proj.description = description
        proj.updated_at = time.time()
        atomic_write_json(self.projects_dir / f"{project_id}.json", proj)
        return proj

    def list_projects(self) -> List[Project]:
        projects = []
        for p in self.projects_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    projects.append(Project(**json.load(f)))
            except Exception:
                continue
        projects.sort(key=lambda pr: pr.updated_at, reverse=True)
        return projects

    def get_project(self, project_id: str) -> Optional[Project]:
        meta_file = self.projects_dir / f"{project_id}.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return Project(**json.load(f))
        except Exception:
            return None

    def add_asset_to_project(self, project_id: str, asset_id: str):
        proj = self.get_project(project_id)
        if proj and asset_id not in proj.asset_ids:
            proj.asset_ids.append(asset_id)
            proj.updated_at = time.time()
            atomic_write_json(self.projects_dir / f"{project_id}.json", proj)

    def add_job_to_project(self, project_id: str, job_id: str):
        proj = self.get_project(project_id)
        if proj and job_id not in proj.job_ids:
            proj.job_ids.append(job_id)
            proj.updated_at = time.time()
            atomic_write_json(self.projects_dir / f"{project_id}.json", proj)

    def delete_project(self, project_id: str) -> bool:
        meta_file = self.projects_dir / f"{project_id}.json"
        if meta_file.exists():
            meta_file.unlink()
            return True
        return False


workspace_store = WorkspaceStore()
