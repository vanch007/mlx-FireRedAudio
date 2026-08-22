"""Persistent asynchronous JobQueue for FireRedAudio Studio."""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from .config import DEFAULT_WORKSPACE_DIR, QUALITY_PRESETS
from .manager import model_manager
from .manager import init_mlx_thread
from .sse import broadcaster
from .workspace import workspace_store, Asset, atomic_write_json

logger = logging.getLogger(__name__)


class JobCreate(BaseModel):
    task: str  # asr | understand | tts | edit_acoustic | edit_semantic | voice_design
    project_id: Optional[str] = None
    preset: Optional[str] = "balanced"  # fast | balanced | high_quality | custom
    params: Dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task: str
    status: str = "queued"  # queued | loading | preprocessing | inferencing | exporting | completed | failed | cancelled | interrupted
    progress: float = 0.0
    current_step: str = "任务已排队等待处理"
    preset: Optional[str] = "balanced"
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_seconds: float = 0.0
    project_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class JobQueue:
    def __init__(self):
        self.jobs_dir = workspace_store.jobs_dir
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running_job_id: Optional[str] = None

    def _save_job(self, job: Job):
        job_file = self.jobs_dir / f"{job.id}.json"
        atomic_write_json(job_file, job)

    def get_job(self, job_id: str) -> Optional[Job]:
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            return None
        try:
            with open(job_file, "r", encoding="utf-8") as f:
                return Job(**json.load(f))
        except Exception:
            return None

    def list_jobs(self, project_id: Optional[str] = None, limit: int = 50) -> List[Job]:
        jobs = []
        for p in self.jobs_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    job = Job(**json.load(f))
                    if project_id is None or job.project_id == project_id:
                        jobs.append(job)
            except Exception:
                continue
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    async def submit_job(self, job_create: JobCreate) -> Job:
        preset_name = job_create.preset or "balanced"
        preset_cfg = QUALITY_PRESETS.get(preset_name, {})
        params = dict(job_create.params)
        for k, v in preset_cfg.items():
            if k not in ("id", "name", "description") and k not in params:
                params[k] = v

        job = Job(
            task=job_create.task,
            project_id=job_create.project_id,
            preset=preset_name,
            params=params,
        )
        self._save_job(job)
        if job.project_id:
            workspace_store.add_job_to_project(job.project_id, job.id)
        await self.queue.put(job.id)
        await broadcaster.broadcast("job_created", job.model_dump())
        return job

    async def retry_job(self, job_id: str) -> Job:
        old_job = self.get_job(job_id)
        if not old_job:
            raise ValueError(f"找不到指定的任务: {job_id}")
        if old_job.status not in ("failed", "interrupted", "cancelled"):
            raise ValueError(f"当前任务状态为 '{old_job.status}'，仅已失败、已中断或已取消的任务支持重新提交。")

        new_job = Job(
            task=old_job.task,
            project_id=old_job.project_id,
            preset=old_job.preset,
            params=dict(old_job.params),
        )
        self._save_job(new_job)
        if new_job.project_id:
            workspace_store.add_job_to_project(new_job.project_id, new_job.id)
        await self.queue.put(new_job.id)
        await broadcaster.broadcast("job_created", new_job.model_dump())
        return new_job

    async def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.current_step = "任务已取消"
            job.completed_at = time.time()
            self._save_job(job)
            await broadcaster.broadcast("job_cancelled", job.model_dump())
            return True
        # Currently running tasks cannot be aborted midway to avoid dirty MLX runtime state
        return False

    async def start(self):
        # Recover stale in-flight jobs to interrupted state & reload queued jobs
        unprocessed_jobs: List[Job] = []
        for p in self.jobs_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    j = Job(**json.load(f))
                    if j.status in ("loading", "preprocessing", "inferencing", "exporting"):
                        j.status = "interrupted"
                        j.current_step = "服务启动/重启，遗留运行中任务已标记为中断。"
                        j.completed_at = time.time()
                        self._save_job(j)
                    elif j.status == "queued":
                        unprocessed_jobs.append(j)
            except Exception:
                continue

        unprocessed_jobs.sort(key=lambda j: j.created_at)
        for j in unprocessed_jobs:
            await self.queue.put(j.id)

        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("JobQueue worker loop started with %d re-queued jobs.", len(unprocessed_jobs))

    async def _update_job_status(self, job: Job, status: str, step: str, progress: float = 0.0):
        job.status = status
        job.current_step = step
        job.progress = progress
        self._save_job(job)
        await broadcaster.broadcast("job_progress", job.model_dump())

    async def _worker_loop(self):
        while True:
            job_id = await self.queue.get()
            job = self.get_job(job_id)
            if not job or job.status == "cancelled":
                self.queue.task_done()
                continue

            self._running_job_id = job.id
            job.started_at = time.time()
            t0 = time.time()

            try:
                await self._update_job_status(job, "loading", "正在准备 MLX 模型运行环境...", 0.1)
                # Offload blocking MLX inference calls to thread pool
                engine = await asyncio.to_thread(model_manager.ensure_ready)

                await self._update_job_status(job, "preprocessing", "正在预处理音频与提示词...", 0.25)
                result = await asyncio.to_thread(self._execute_inference, engine, job)

                job.latency_seconds = round(time.time() - t0, 4)
                job.completed_at = time.time()
                audio_dur = (result or {}).get("duration_s") or (result or {}).get("audio_duration_s") or 0.0
                if audio_dur > 0 and job.latency_seconds > 0:
                    rtf = round(job.latency_seconds / audio_dur, 4)
                    result["rtf"] = rtf
                    result["duration_s"] = round(audio_dur, 2)
                job.result = result
                job.status = "completed"
                job.progress = 1.0
                rtf_info = f" | RTF: {result['rtf']}" if result and "rtf" in result else ""
                job.current_step = f"推理完成 (耗时: {job.latency_seconds}s{rtf_info})"
                self._save_job(job)
                await broadcaster.broadcast("job_completed", job.model_dump())

            except Exception as e:
                logger.error("Job %s failed: %s", job.id, e, exc_info=True)
                job.latency_seconds = round(time.time() - t0, 4)
                job.completed_at = time.time()
                job.status = "failed"
                job.error = str(e)
                job.current_step = f"任务执行失败: {e}"
                self._save_job(job)
                await broadcaster.broadcast("job_failed", job.model_dump())
            finally:
                self._running_job_id = None
                self.queue.task_done()
                # Clean temporary GPU metal buffers
                model_manager.clear_cache()

    def _execute_inference(self, engine, job: Job) -> Dict[str, Any]:
        init_mlx_thread()
        task = job.task
        p = job.params

        # 1. ASR (Speech Recognition)
        if task == "asr":
            audio_ids = p.get("audio_asset_ids", [])
            if isinstance(audio_ids, str):
                audio_ids = [audio_ids]
            audio_paths = []
            total_dur = 0.0
            for aid in audio_ids:
                asset = workspace_store.get_asset(aid)
                if not asset or not os.path.exists(asset.file_path):
                    raise FileNotFoundError(f"找不到音频素材: {aid}")
                audio_paths.append(asset.file_path)
                total_dur += asset.duration

            prompt = p.get("prompt", "Transcribe speech to text.")
            max_new_tokens = int(p.get("max_new_tokens", 300))
            num_beams = int(p.get("asr_beam_size", 4))
            temp = float(p.get("temperature", 0.0))
            top_k = int(p.get("top_k", 20))
            top_p = float(p.get("top_p", 0.8))
            out = engine.understand(
                audio_paths=audio_paths,
                prompt=prompt,
                task="asr",
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
            )
            if not out.answer.strip():
                raise ValueError("语音识别未返回有效文本内容。")
            return {
                "transcript": out.answer,
                "num_audios": len(audio_paths),
                "audio_duration_s": round(total_dur, 2),
            }

        # 2. Audio Understanding / QA (with optional CoT thinking)
        elif task == "understand":
            audio_ids = p.get("audio_asset_ids", [])
            if isinstance(audio_ids, str):
                audio_ids = [audio_ids]
            audio_paths = []
            total_dur = 0.0
            for aid in audio_ids:
                asset = workspace_store.get_asset(aid)
                if not asset or not os.path.exists(asset.file_path):
                    raise FileNotFoundError(f"找不到音频素材: {aid}")
                audio_paths.append(asset.file_path)
                total_dur += asset.duration

            prompt = p.get("prompt", "这个音频中有几个说话人？")
            enable_thinking = bool(p.get("enable_thinking", False))
            max_new_tokens = int(p.get("max_new_tokens", 1024 if enable_thinking else 300))
            out = engine.understand(
                audio_paths=audio_paths,
                prompt=prompt,
                task="understand",
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
            if not out.answer.strip():
                raise ValueError("音频问答未返回有效回答。")
            return {
                "answer": out.answer,
                "reasoning": out.reasoning,
                "enable_thinking": enable_thinking,
                "audio_duration_s": round(total_dur, 2),
            }

        # 3. Zero-shot TTS (Voice Cloning)
        elif task == "tts":
            prompt_audio_id = p.get("prompt_audio_asset_id")
            prompt_text = p.get("prompt_text", "").strip()
            target_text = p.get("target_text", "").strip()
            language = p.get("language", "zh")

            if not prompt_text or not target_text:
                raise ValueError("TTS 声音克隆必须提供参考文本和目标合成文本。")

            asset = workspace_store.get_asset(prompt_audio_id) if prompt_audio_id else None
            if not asset or not os.path.exists(asset.file_path):
                raise FileNotFoundError(f"找不到参考音频素材: {prompt_audio_id}")

            n_timesteps = int(p.get("n_timesteps", 4))
            inference_cfg = float(p.get("inference_cfg", 2.0))
            max_new_audio_steps = int(p.get("max_new_audio_steps", 750))

            res = engine.tts(
                prompt_text=prompt_text,
                prompt_audio=asset.file_path,
                target_text=target_text,
                language=language,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                min_new_audio_steps=0,
                max_new_audio_steps=max_new_audio_steps,
            )
            return self._save_generated_audio(res.audio, res.sample_rate, f"tts_{job.id}.wav", job.project_id, target_text)

        # 4. Acoustic Speech Editing
        elif task == "edit_acoustic":
            audio_id = p.get("audio_asset_id")
            asset = workspace_store.get_asset(audio_id) if audio_id else None
            if not asset or not os.path.exists(asset.file_path):
                raise FileNotFoundError(f"找不到编辑原音频素材: {audio_id}")

            speed = p.get("speed")
            pitch = p.get("pitch")
            volume = p.get("volume")
            mode = p.get("mode")
            custom_instruction = p.get("instruction")

            if custom_instruction:
                instruction = custom_instruction
            elif mode == "speed" and speed is not None:
                instruction = f"adjust the speed to {float(speed):.1f}"
            elif mode == "pitch" and pitch is not None:
                val = int(pitch)
                instruction = f"shift the pitch by {val} step{'s' if abs(val) != 1 else ''}"
            elif mode == "volume" and volume is not None:
                instruction = f"adjust the volume to {float(volume):.1f}"
            elif speed is not None:
                instruction = f"adjust the speed to {float(speed):.1f}"
            elif pitch is not None:
                val = int(pitch)
                instruction = f"shift the pitch by {val} step{'s' if abs(val) != 1 else ''}"
            elif volume is not None:
                instruction = f"adjust the volume to {float(volume):.1f}"
            else:
                instruction = "adjust the speed to 1.0"

            n_timesteps = int(p.get("n_timesteps", 4))
            inference_cfg = float(p.get("inference_cfg", 2.0))
            max_new_audio_steps = int(p.get("max_new_audio_steps", 750))

            res = engine.edit(
                audio_path=asset.file_path,
                instruction=instruction,
                edit_type="acoustic",
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                min_new_audio_steps=0,
                max_new_audio_steps=max_new_audio_steps,
            )
            return self._save_generated_audio(res.audio, res.sample_rate, f"edit_acoustic_{job.id}.wav", job.project_id, instruction)

        # 5. Semantic Speech Editing
        elif task == "edit_semantic":
            audio_id = p.get("audio_asset_id")
            asset = workspace_store.get_asset(audio_id) if audio_id else None
            if not asset or not os.path.exists(asset.file_path):
                raise FileNotFoundError(f"找不到语义编辑原音频: {audio_id}")

            instruction = p.get("instruction", "").strip()
            if not instruction:
                raise ValueError("语义编辑需要指定修改指令 (如 delete '...' 或 replace 'A' with 'B')。")

            n_timesteps = int(p.get("n_timesteps", 4))
            inference_cfg = float(p.get("inference_cfg", 2.0))
            max_new_audio_steps = int(p.get("max_new_audio_steps", 750))
            max_new_text_tokens = max(int(p.get("max_new_text_tokens", 512)), 512)

            res = engine.edit(
                audio_path=asset.file_path,
                instruction=instruction,
                edit_type="semantic",
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                min_new_audio_steps=0,
                max_new_audio_steps=max_new_audio_steps,
                max_new_text_tokens=max_new_text_tokens,
            )
            if not res.text or not res.text.strip():
                raise ValueError("语义编辑任务未生成改写文本。")
            data = self._save_generated_audio(res.audio, res.sample_rate, f"edit_semantic_{job.id}.wav", job.project_id, res.text or instruction)
            data["rewritten_text"] = res.text
            return data

        # 6. Voice Design
        elif task == "voice_design":
            instruction = p.get("instruction", "").strip()
            text = p.get("text", "").strip()
            if not instruction or not text:
                raise ValueError("音色设计需要提供音色描述指令和目标合成文本。")

            n_timesteps = int(p.get("n_timesteps", 4))
            inference_cfg = float(p.get("inference_cfg", 2.0))
            max_new_audio_steps = int(p.get("max_new_audio_steps", 750))
            max_new_text_tokens = max(int(p.get("max_new_text_tokens", 512)), 512)

            res = engine.voice_design(
                instruction=instruction,
                text=text,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                min_new_audio_steps=0,
                max_new_audio_steps=max_new_audio_steps,
                max_new_text_tokens=max_new_text_tokens,
            )
            data = self._save_generated_audio(res.audio, res.sample_rate, f"voice_design_{job.id}.wav", job.project_id, text)
            data["timbre_description"] = res.text
            return data

        else:
            raise NotImplementedError(f"未知的任务类型: {task}")

    def _save_generated_audio(self, audio: np.ndarray, sample_rate: int, filename: str, project_id: Optional[str], text_note: Optional[str]) -> Dict[str, Any]:
        if len(audio) == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
            raise ValueError("模型生成了静音、空或非有限数值的异常音频。")

        # Write to temporary file first and re-verify integrity
        dest_path = workspace_store.outputs_dir / filename
        tmp_dest = workspace_store.outputs_dir / f"tmp_{uuid.uuid4().hex[:6]}_{filename}"
        sf.write(str(tmp_dest), audio, sample_rate, subtype="PCM_16")

        # Re-read verification per Spec Section 10
        re_audio, re_sr = sf.read(str(tmp_dest))
        if len(re_audio) == 0 or re_sr != sample_rate or not np.isfinite(re_audio).all():
            tmp_dest.unlink(missing_ok=True)
            raise ValueError(f"生成音频复核校验失败（采样率: {re_sr}, 期望: {sample_rate}）。")

        tmp_dest.replace(dest_path)

        dur = round(len(audio) / sample_rate, 2)
        asset_id = str(uuid.uuid4())[:8]
        dest_filename = f"{asset_id}_{filename}"
        asset_path = workspace_store.assets_dir / dest_filename
        shutil.copyfile(str(dest_path), str(asset_path))

        asset = Asset(
            id=asset_id,
            name=filename,
            file_path=str(asset_path),
            media_url=f"/api/v1/media/assets/{dest_filename}",
            file_size=asset_path.stat().st_size,
            duration=dur,
            sample_rate=sample_rate,
            channels=1,
            source="generation",
            project_id=project_id,
        )
        atomic_write_json(workspace_store.assets_dir / f"{asset_id}.json", asset)

        if project_id:
            workspace_store.add_asset_to_project(project_id, asset_id)

        return {
            "asset_id": asset.id,
            "media_url": asset.media_url,
            "duration_s": dur,
            "sample_rate": sample_rate,
            "text": text_note,
        }


job_queue = JobQueue()
