#!/usr/bin/env python3
"""Comprehensive benchmark for mlx-FireRedAudio."""

import json
import time
import os
import sys
import soundfile as sf
import numpy as np
from fireredaudio_mlx import FireRedAudioInference

def run_bench():
    print("=" * 75, flush=True)
    print("mlx-FireRedAudio FULL CAPABILITY REAL MLX BENCHMARK", flush=True)
    print("=" * 75, flush=True)

    os.makedirs("outputs", exist_ok=True)
    results = {"profile": "m3_max_balanced", "flow_timesteps": 4}

    t0 = time.time()
    engine = FireRedAudioInference(model_path="models/FireRedAudio")
    load_time = engine.load_time
    print(f"MLX Model Loaded in {load_time:.4f}s", flush=True)
    results["model_load_time_seconds"] = round(load_time, 4)

    # 1. ASR (Speech Recognition)
    asr_file = "assets/examples/asr_zh_fleurs.wav"
    d_asr, sr_asr = sf.read(asr_file)
    dur_asr = len(d_asr) / sr_asr
    print(f"\n[1/6] Testing ASR ({asr_file}, duration: {dur_asr:.2f}s)...", flush=True)
    t0 = time.time()
    res_asr = engine.understand(asr_file, task="asr", max_new_tokens=50)
    t_asr = time.time() - t0
    rtf_asr = t_asr / dur_asr
    print(f"Transcript: {res_asr.answer!r}", flush=True)
    print(f"Latency: {t_asr:.4f}s | RTF: {rtf_asr:.4f}", flush=True)
    results["asr"] = {"latency_s": round(t_asr, 4), "rtf": round(rtf_asr, 4), "transcript": res_asr.answer}
    if not res_asr.answer.strip():
        raise AssertionError("ASR returned an empty transcript")

    # 2. Audio Understanding / QA with Thinking
    qa_file = "assets/examples/two_speakers.wav"
    d_qa, sr_qa = sf.read(qa_file)
    dur_qa = len(d_qa) / sr_qa
    print(f"\n[2/6] Testing Audio QA with Thinking ({qa_file}, duration: {dur_qa:.2f}s)...", flush=True)
    t0 = time.time()
    res_qa = engine.understand(qa_file, prompt="这个音频中有几个说话人？", task="understand", enable_thinking=True, max_new_tokens=256)
    t_qa = time.time() - t0
    print(f"Answer: {res_qa.answer!r}", flush=True)
    if res_qa.reasoning:
        print(f"Reasoning: {res_qa.reasoning!r}", flush=True)
    print(f"Latency: {t_qa:.4f}s", flush=True)
    results["qa"] = {"latency_s": round(t_qa, 4), "answer": res_qa.answer, "reasoning": res_qa.reasoning}
    if not res_qa.answer.strip():
        raise AssertionError("Audio QA returned an empty answer")

    # 3. Zero-shot TTS Voice Cloning
    print("\n[3/6] Testing Zero-shot TTS Voice Cloning (RedAE -> Backbone -> DiT -> RedAE Decoder)...", flush=True)
    prompt_audio = "assets/examples/tts_zh_prompt.wav"
    prompt_text = "收到你的来信，我很高兴。"
    target_text = "你好，这是使用独立 mlx-FireRedAudio 项目在苹果芯片上运行的原生声音克隆测试。"
    t0 = time.time()
    res_tts = engine.tts(
        prompt_text=prompt_text,
        prompt_audio=prompt_audio,
        target_text=target_text,
        language="zh",
        n_timesteps=4,
        min_new_audio_steps=0,
        max_new_audio_steps=100,
    )
    t_tts = time.time() - t0
    dur_tts = len(res_tts.audio) / res_tts.sample_rate
    if dur_tts <= 0 or not np.isfinite(res_tts.audio).all() or np.max(np.abs(res_tts.audio)) == 0:
        raise AssertionError("TTS returned empty, non-finite, or silent audio")
    out_tts = "outputs/benchmark_tts.wav"
    sf.write(out_tts, res_tts.audio, res_tts.sample_rate, subtype="PCM_16")
    rtf_tts = t_tts / dur_tts
    print(f"Saved audio: {out_tts} (duration: {dur_tts:.2f}s)", flush=True)
    print(f"Latency: {t_tts:.4f}s | RTF: {rtf_tts:.4f}", flush=True)
    results["tts"] = {"latency_s": round(t_tts, 4), "rtf": round(rtf_tts, 4), "output_audio": out_tts, "duration_s": round(dur_tts, 2)}

    # 4. Speech Editing (Acoustic / Semantic)
    print("\n[4/6] Testing Speech Editing (Acoustic modification)...", flush=True)
    edit_ref = "assets/examples/edit_acoustic_zh_ref.wav"
    t0 = time.time()
    res_edit = engine.edit(
        audio_path=edit_ref,
        instruction="adjust the speed to 0.5",
        edit_type="acoustic",
        n_timesteps=4,
        min_new_audio_steps=0,
        max_new_audio_steps=100,
    )
    t_edit = time.time() - t0
    dur_edit = len(res_edit.audio) / res_edit.sample_rate
    if dur_edit <= 0 or not np.isfinite(res_edit.audio).all() or np.max(np.abs(res_edit.audio)) == 0:
        raise AssertionError("Speech editing returned empty, non-finite, or silent audio")
    out_edit = "outputs/benchmark_edit_slow.wav"
    sf.write(out_edit, res_edit.audio, res_edit.sample_rate, subtype="PCM_16")
    print(f"Saved audio: {out_edit} (duration: {dur_edit:.2f}s)", flush=True)
    print(f"Latency: {t_edit:.4f}s", flush=True)
    results["edit"] = {"latency_s": round(t_edit, 4), "output_audio": out_edit, "duration_s": round(dur_edit, 2)}

    # 4b. Semantic speech editing
    semantic_ref = "assets/examples/edit_semantic_zh_ref.wav"
    print("\n[5/6] Testing Speech Editing (Semantic deletion)...", flush=True)
    t0 = time.time()
    res_semantic = engine.edit(
        audio_path=semantic_ref,
        instruction="delete '比普通的茶叶要'",
        edit_type="semantic",
        n_timesteps=4,
        min_new_audio_steps=0,
        max_new_audio_steps=100,
        max_new_text_tokens=256,
    )
    t_semantic = time.time() - t0
    dur_semantic = len(res_semantic.audio) / res_semantic.sample_rate
    out_semantic = "outputs/benchmark_edit_semantic.wav"
    sf.write(out_semantic, res_semantic.audio, res_semantic.sample_rate, subtype="PCM_16")
    if not res_semantic.text or dur_semantic <= 0 or not np.isfinite(res_semantic.audio).all():
        raise AssertionError("Semantic editing did not return rewritten text and valid audio")
    results["semantic_edit"] = {
        "latency_s": round(t_semantic, 4),
        "rewritten_text": res_semantic.text,
        "output_audio": out_semantic,
        "duration_s": round(dur_semantic, 2),
    }

    # 5. Voice Design (Timbre guided synthesis)
    print("\n[6/6] Testing Voice Design (Timbre guided synthesis)...", flush=True)
    t0 = time.time()
    res_vd = engine.voice_design(
        instruction="温柔知性的广播女主播声音",
        text="欢迎收听今日新闻，我们将为您带来最新的科技前沿报道。",
        n_timesteps=4,
        min_new_audio_steps=0,
        max_new_audio_steps=100,
    )
    t_vd = time.time() - t0
    dur_vd = len(res_vd.audio) / res_vd.sample_rate if len(res_vd.audio) > 0 else 0.0
    if dur_vd <= 0 or not np.isfinite(res_vd.audio).all() or np.max(np.abs(res_vd.audio)) == 0:
        raise AssertionError("Voice design returned empty, non-finite, or silent audio")
    out_vd = "outputs/benchmark_voice_design.wav"
    if len(res_vd.audio) > 0:
        sf.write(out_vd, res_vd.audio, res_vd.sample_rate, subtype="PCM_16")
    print(f"Saved audio: {out_vd} (duration: {dur_vd:.2f}s)", flush=True)
    print(f"Latency: {t_vd:.4f}s", flush=True)
    results["voice_design"] = {"latency_s": round(t_vd, 4), "output_audio": out_vd, "duration_s": round(dur_vd, 2)}

    print("\n" + "=" * 75, flush=True)
    results["status"] = "pass"
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("ALL 6 CAPABILITY CHECKS PASSED WITH REAL MLX INFERENCE!", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_bench()
