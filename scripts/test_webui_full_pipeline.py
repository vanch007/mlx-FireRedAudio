#!/usr/bin/env python3
"""Synchronous in-process test of WebUI JobQueue + ModelManager + API router."""

import asyncio
import json
import time
import unittest
import soundfile as sf
import numpy as np
from pathlib import Path

from fireredaudio_mlx.webui.manager import model_manager
from fireredaudio_mlx.webui.workspace import workspace_store
from fireredaudio_mlx.webui.jobs import job_queue, JobCreate

async def async_test_pipeline():
    print("=" * 70, flush=True)
    print("Testing WebUI Full Pipeline (JobQueue + ModelManager + Storage)", flush=True)
    print("=" * 70, flush=True)

    # 1. Start queue worker
    await job_queue.start()

    # 2. Ensure model is ready
    print("\n[1/4] Ensuring model is ready...", flush=True)
    t0 = time.time()
    engine = await asyncio.to_thread(model_manager.ensure_ready)
    print(f"Model ready in {time.time() - t0:.4f}s", flush=True)

    # 3. Create Project & Upload Asset
    print("\n[2/4] Creating project & asset...", flush=True)
    proj = workspace_store.create_project("WebUI Pipeline Verification", "E2E automated test")
    wav_path = "assets/examples/asr_zh_fleurs.wav"
    with open(wav_path, "rb") as f:
        content = f.read()
    asset = workspace_store.save_uploaded_asset("test_fleurs.wav", content, project_id=proj.id)
    print(f"Project #{proj.id} created with Asset #{asset.id} (duration: {asset.duration}s)", flush=True)

    # 4. Submit & Wait for ASR Job
    print("\n[3/4] Submitting ASR job to queue...", flush=True)
    job_asr = await job_queue.submit_job(JobCreate(
        task="asr",
        project_id=proj.id,
        params={"audio_asset_ids": [asset.id], "max_new_tokens": 12},
    ))
    print(f"Job #{job_asr.id} queued, waiting for worker completion...", flush=True)

    t0 = time.time()
    while True:
        await asyncio.sleep(1.0)
        j = job_queue.get_job(job_asr.id)
        print(f"  status: {j.status} | progress: {j.progress*100:.0f}% | step: {j.current_step}", flush=True)
        if j.status == "completed":
            print(f"✓ ASR Job COMPLETED! Transcript: '{j.result['transcript']}' (Latency: {j.latency_seconds}s)", flush=True)
            assert "浪漫主义" in j.result["transcript"], "ASR output mismatch"
            break
        elif j.status == "failed":
            raise AssertionError(f"ASR Job failed: {j.error}")
        if time.time() - t0 > 90:
            raise TimeoutError("ASR Job timed out")

    # 5. Submit TTS Job with Fast Preset
    print("\n[4/4] Submitting TTS Voice Cloning job to queue...", flush=True)
    prompt_wav = "assets/examples/tts_zh_prompt.wav"
    with open(prompt_wav, "rb") as f:
        prompt_content = f.read()
    prompt_asset = workspace_store.save_uploaded_asset("prompt.wav", prompt_content, project_id=proj.id)

    job_tts = await job_queue.submit_job(JobCreate(
        task="tts",
        project_id=proj.id,
        params={
            "prompt_audio_asset_id": prompt_asset.id,
            "prompt_text": "同时，他强调微调要科学有序。",
            "target_text": "你好，这是 WebUI 语音克隆全流程测试。",
            "n_timesteps": 4,
            "max_new_audio_steps": 30,
        },
    ))

    t0 = time.time()
    while True:
        await asyncio.sleep(1.0)
        j = job_queue.get_job(job_tts.id)
        print(f"  status: {j.status} | progress: {j.progress*100:.0f}% | step: {j.current_step}", flush=True)
        if j.status == "completed":
            print(f"✓ TTS Job COMPLETED! Output: {j.result['media_url']} (Duration: {j.result['duration_s']}s, Latency: {j.latency_seconds}s)", flush=True)
            assert j.result["duration_s"] > 0, "TTS audio duration must be positive"
            break
        elif j.status == "failed":
            raise AssertionError(f"TTS Job failed: {j.error}")
        if time.time() - t0 > 90:
            raise TimeoutError("TTS Job timed out")

    print("\n" + "=" * 70, flush=True)
    print("🎉 ALL WEBUI ASYNC QUEUE & INFERENCE PIPELINES VERIFIED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    asyncio.run(async_test_pipeline())
