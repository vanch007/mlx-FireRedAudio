import json
import time
import soundfile as sf
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from fireredaudio_mlx import FireRedAudioInference
from fireredaudio_mlx.models.modeling import FireRedAudioModel

def test_optimizations():
    print("=" * 75)
    print("FireRedAudio MLX: Testing Optimization & 4-bit/8-bit Quantization")
    print("=" * 75)

    # 1. Baseline BF16
    print("[1/3] Loading Baseline BF16 Engine...")
    t0 = time.time()
    engine_bf16 = FireRedAudioInference(model_path="models/FireRedAudio")
    mem_bf16 = mx.metal.get_active_memory() / (1024**3)
    print(f"BF16 Engine loaded in {time.time() - t0:.2f}s | Active Memory: {mem_bf16:.2f} GB")

    # Benchmark ASR BF16 (greedy beam=1 for fast comparison)
    t0 = time.time()
    res_asr_bf16 = engine_bf16.understand("assets/examples/asr_zh_fleurs.wav", task="asr", max_new_tokens=40, num_beams=1)
    t_asr_bf16 = time.time() - t0
    dur_asr = len(sf.read("assets/examples/asr_zh_fleurs.wav")[0]) / 16000.0
    print(f"ASR BF16 (1-beam) Latency: {t_asr_bf16:.2f}s | RTF: {t_asr_bf16 / dur_asr:.4f} | Transcript: '{res_asr_bf16.answer}'")

    # 2. Test On-The-Fly 8-bit Quantization
    print("[2/3] Quantizing Backbone to 8-bit in-place...")
    def quant_pred(path, m):
        if not isinstance(m, nn.Linear):
            return False
        if "red_vae" in path or "vae_decoder" in path or "audio_encoder" in path:
            return False
        if "backbone_llm" in path:
            return True
        return False

    t0 = time.time()
    nn.quantize(engine_bf16.model, group_size=64, bits=8, class_predicate=quant_pred)
    mx.eval(engine_bf16.model.parameters())
    mx.metal.clear_cache()
    mem_8bit = mx.metal.get_active_memory() / (1024**3)
    print(f"8-bit Quantization completed in {time.time() - t0:.2f}s | Active Memory: {mem_8bit:.2f} GB")

    # Benchmark ASR 8-bit
    t0 = time.time()
    res_asr_8bit = engine_bf16.understand("assets/examples/asr_zh_fleurs.wav", task="asr", max_new_tokens=40, num_beams=1)
    t_asr_8bit = time.time() - t0
    print(f"ASR 8-bit (1-beam) Latency: {t_asr_8bit:.2f}s | RTF: {t_asr_8bit / dur_asr:.4f} | Transcript: '{res_asr_8bit.answer}'")

    # Benchmark TTS 8-bit
    t0 = time.time()
    res_tts_8bit = engine_bf16.tts(
        prompt_text="收到你的来信，我很高兴。",
        prompt_audio="assets/examples/tts_zh_prompt.wav",
        target_text="你好，这是八比特量化语音合成测试。",
        n_timesteps=4,
        max_new_audio_steps=30,
    )
    t_tts_8bit = time.time() - t0
    dur_tts = len(res_tts_8bit.audio) / res_tts_8bit.sample_rate
    print(f"TTS 8-bit Latency: {t_tts_8bit:.2f}s | Output: {dur_tts:.2f}s | RTF: {t_tts_8bit / dur_tts:.4f}")

    # 3. Test On-The-Fly 4-bit Quantization
    print("[3/3] Testing fresh 4-bit Quantized Engine...")
    engine_4bit = FireRedAudioInference(model_path="models/FireRedAudio")
    nn.quantize(engine_4bit.model, group_size=64, bits=4, class_predicate=quant_pred)
    mx.eval(engine_4bit.model.parameters())
    mx.metal.clear_cache()
    mem_4bit = mx.metal.get_active_memory() / (1024**3)
    print(f"4-bit Engine Ready | Active Memory: {mem_4bit:.2f} GB")

    t0 = time.time()
    res_asr_4bit = engine_4bit.understand("assets/examples/asr_zh_fleurs.wav", task="asr", max_new_tokens=40, num_beams=1)
    t_asr_4bit = time.time() - t0
    print(f"ASR 4-bit (1-beam) Latency: {t_asr_4bit:.2f}s | RTF: {t_asr_4bit / dur_asr:.4f} | Transcript: '{res_asr_4bit.answer}'")

    t0 = time.time()
    res_tts_4bit = engine_4bit.tts(
        prompt_text="收到你的来信，我很高兴。",
        prompt_audio="assets/examples/tts_zh_prompt.wav",
        target_text="你好，这是四比特量化语音合成测试。",
        n_timesteps=4,
        max_new_audio_steps=30,
    )
    t_tts_4bit = time.time() - t0
    dur_tts4 = len(res_tts_4bit.audio) / res_tts_4bit.sample_rate
    print(f"TTS 4-bit Latency: {t_tts_4bit:.2f}s | Output: {dur_tts4:.2f}s | RTF: {t_tts_4bit / dur_tts4:.4f}")

    print("=" * 75)
    print("QUANTIZATION PERFORMANCE COMPARISON SUMMARY:")
    print(f"  - BF16:  Memory = {mem_bf16:.2f} GB | ASR Latency = {t_asr_bf16:.2f}s (RTF {t_asr_bf16/dur_asr:.2f})")
    print(f"  - 8-bit: Memory = {mem_8bit:.2f} GB | ASR Latency = {t_asr_8bit:.2f}s (RTF {t_asr_8bit/dur_asr:.2f}) | TTS RTF = {t_tts_8bit/dur_tts:.2f}")
    print(f"  - 4-bit: Memory = {mem_4bit:.2f} GB | ASR Latency = {t_asr_4bit:.2f}s (RTF {t_asr_4bit/dur_asr:.2f}) | TTS RTF = {t_tts_4bit/dur_tts4:.2f}")
    print("=" * 75)

if __name__ == "__main__":
    test_optimizations()
