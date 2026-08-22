#!/usr/bin/env python3
"""10-task consecutive stability and single-instance memory test."""

import asyncio
import json
import time
import soundfile as sf
import numpy as np
from pathlib import Path

from fireredaudio_mlx.webui.manager import model_manager
from fireredaudio_mlx.webui.workspace import workspace_store
from fireredaudio_mlx.webui.jobs import job_queue, JobCreate


async def wait_job_completed(job_id: str, timeout: float = 120.0):
    t0 = time.time()
    while True:
        await asyncio.sleep(0.5)
        j = job_queue.get_job(job_id)
        if not j:
            raise RuntimeError(f"Job {job_id} not found")
        if j.status == "completed":
            return j
        elif j.status in ("failed", "cancelled", "interrupted"):
            raise AssertionError(f"Job {job_id} ended with status '{j.status}': {j.error}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Job {job_id} timed out after {timeout}s (step: {j.current_step})")


async def run_stress_test():
    print("=" * 75, flush=True)
    print("WebUI 10-Task Consecutive Stability & Single-Instance Memory Test", flush=True)
    print("=" * 75, flush=True)

    # 1. Start queue worker
    await job_queue.start()

    # 2. Ensure model is ready
    print("Ensuring model is loaded...", flush=True)
    t0 = time.time()
    engine = await asyncio.to_thread(model_manager.ensure_ready)
    initial_mem = model_manager.get_memory_info()
    print(f"Model ready in {time.time() - t0:.4f}s | Active Metal: {initial_mem['active_memory_gb']} GB", flush=True)

    # Create project & assets
    proj = workspace_store.create_project("Stress Test Suite", "10-Task Consecutive Verification")

    # Upload assets
    with open("assets/examples/asr_zh_fleurs.wav", "rb") as f:
        asr_asset = workspace_store.save_uploaded_asset("fleurs.wav", f.read(), project_id=proj.id)
    with open("assets/examples/two_speakers.wav", "rb") as f:
        qa_asset = workspace_store.save_uploaded_asset("speakers.wav", f.read(), project_id=proj.id)
    with open("assets/examples/tts_zh_prompt.wav", "rb") as f:
        tts_asset = workspace_store.save_uploaded_asset("tts_prompt.wav", f.read(), project_id=proj.id)
    with open("assets/examples/edit_acoustic_zh_ref.wav", "rb") as f:
        edit_ac_asset = workspace_store.save_uploaded_asset("edit_ac.wav", f.read(), project_id=proj.id)
    with open("assets/examples/edit_semantic_zh_ref.wav", "rb") as f:
        edit_sem_asset = workspace_store.save_uploaded_asset("edit_sem.wav", f.read(), project_id=proj.id)

    tasks_spec = [
        ("asr", {"audio_asset_ids": [asr_asset.id], "max_new_tokens": 15}),
        ("understand", {"audio_asset_ids": [qa_asset.id], "prompt": "这个音频中有几个说话人？", "enable_thinking": True, "max_new_tokens": 30}),
        ("tts", {"prompt_audio_asset_id": tts_asset.id, "prompt_text": "收到你的来信，我很高兴。", "target_text": "第一条连续语音合成测试。", "n_timesteps": 4, "max_new_audio_steps": 40}),
        ("edit_acoustic", {"audio_asset_id": edit_ac_asset.id, "mode": "speed", "speed": 0.5, "n_timesteps": 4, "max_new_audio_steps": 40}),
        ("edit_semantic", {"audio_asset_id": edit_sem_asset.id, "instruction": "delete '比普通的茶叶要'", "n_timesteps": 4, "max_new_audio_steps": 60, "max_new_text_tokens": 256}),
        ("voice_design", {"instruction": "温柔知性的广播女主播声音", "text": "音色设计稳定性测试。", "n_timesteps": 4, "max_new_audio_steps": 60, "max_new_text_tokens": 256}),
        ("asr", {"audio_asset_ids": [asr_asset.id], "max_new_tokens": 15}),
        ("understand", {"audio_asset_ids": [qa_asset.id], "prompt": "这个音频中有几个人说话？", "enable_thinking": False, "max_new_tokens": 20}),
        ("tts", {"prompt_audio_asset_id": tts_asset.id, "prompt_text": "收到你的来信，我很高兴。", "target_text": "第二条连续语音克隆测试。", "n_timesteps": 4, "max_new_audio_steps": 40}),
        ("edit_acoustic", {"audio_asset_id": edit_ac_asset.id, "mode": "pitch", "pitch": 2, "n_timesteps": 4, "max_new_audio_steps": 40}),
    ]

    results_log = []
    engine_id_initial = id(model_manager.engine)

    for idx, (task_type, params) in enumerate(tasks_spec, start=1):
        print(f"[{idx}/10] Submitting Task: {task_type.upper()}...", flush=True)
        t_start = time.time()
        job = await job_queue.submit_job(JobCreate(
            task=task_type,
            project_id=proj.id,
            params=params,
            preset="fast",
        ))
        completed_job = await wait_job_completed(job.id)
        mem = model_manager.get_memory_info()

        assert id(model_manager.engine) == engine_id_initial, "Engine instance was recreated unexpectedly!"

        summary = {
            "index": idx,
            "task": task_type,
            "job_id": job.id,
            "latency_s": completed_job.latency_seconds,
            "active_memory_gb": mem["active_memory_gb"],
            "peak_memory_gb": mem["peak_memory_gb"],
        }
        results_log.append(summary)
        print(f"  ✓ [{idx}/10] {task_type} COMPLETED in {completed_job.latency_seconds}s (Active: {mem['active_memory_gb']} GB, Peak: {mem['peak_memory_gb']} GB)", flush=True)

    print("=" * 75, flush=True)
    print("ALL 10 CONSECUTIVE TASKS COMPLETED SUCCESSFULLY!", flush=True)
    print("Single Model Engine Instance Preserved Throughout Execution.")
    print("=" * 75, flush=True)

    with open("workspace/stress_test_report.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(run_stress_test())
