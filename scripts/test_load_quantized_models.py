import json
import time
import soundfile as sf
import mlx.core as mx

from fireredaudio_mlx import FireRedAudioInference

def test_quantized_models_from_disk():
    print("=" * 75)
    print("Testing Pre-Quantized 8-bit & 4-bit Standalone Models From Disk")
    print("=" * 75)

    # 1. Test 8-bit Standalone Model
    print("[1/2] Loading models/FireRedAudio-8bit directly from disk...")
    t0 = time.time()
    engine_8bit = FireRedAudioInference(model_path="models/FireRedAudio-8bit")
    mem_8bit = mx.get_active_memory() / (1024**3)
    print(f"✓ 8-bit Model Loaded in {time.time() - t0:.2f}s | Active Memory: {mem_8bit:.2f} GB")

    # Run ASR
    t0 = time.time()
    res_asr = engine_8bit.understand("assets/examples/asr_zh_fleurs.wav", task="asr", max_new_tokens=40, num_beams=1)
    t_asr = time.time() - t0
    dur_asr = len(sf.read("assets/examples/asr_zh_fleurs.wav")[0]) / 16000.0
    print(f"✓ ASR 8-bit Latency: {t_asr:.2f}s | RTF: {t_asr / dur_asr:.4f} | Transcript: '{res_asr.answer}'")
    assert "浪漫主义" in res_asr.answer, "ASR transcript mismatch"

    # Run TTS
    t0 = time.time()
    res_tts = engine_8bit.tts(
        prompt_text="收到你的来信，我很高兴。",
        prompt_audio="assets/examples/tts_zh_prompt.wav",
        target_text="你好，这是直接从磁盘加载的八比特量化模型语音合成。",
        n_timesteps=4,
        max_new_audio_steps=30,
    )
    t_tts = time.time() - t0
    dur_tts = len(res_tts.audio) / res_tts.sample_rate
    print(f"✓ TTS 8-bit Latency: {t_tts:.2f}s | Output: {dur_tts:.2f}s | RTF: {t_tts / dur_tts:.4f}")
    assert dur_tts > 0, "TTS output audio empty"

    # 2. Test 4-bit Standalone Model
    print("[2/2] Loading models/FireRedAudio-4bit directly from disk...")
    del engine_8bit
    mx.clear_cache()
    t0 = time.time()
    engine_4bit = FireRedAudioInference(model_path="models/FireRedAudio-4bit")
    mem_4bit = mx.get_active_memory() / (1024**3)
    print(f"✓ 4-bit Model Loaded in {time.time() - t0:.2f}s | Active Memory: {mem_4bit:.2f} GB")

    t0 = time.time()
    res_asr_4 = engine_4bit.understand("assets/examples/asr_zh_fleurs.wav", task="asr", max_new_tokens=40, num_beams=1)
    t_asr_4 = time.time() - t0
    print(f"✓ ASR 4-bit Latency: {t_asr_4:.2f}s | RTF: {t_asr_4 / dur_asr:.4f} | Transcript: '{res_asr_4.answer}'")
    assert "浪漫主义" in res_asr_4.answer, "ASR transcript mismatch"

    t0 = time.time()
    res_tts_4 = engine_4bit.tts(
        prompt_text="收到你的来信，我很高兴。",
        prompt_audio="assets/examples/tts_zh_prompt.wav",
        target_text="你好，这是直接从磁盘加载的四比特量化模型语音合成。",
        n_timesteps=4,
        max_new_audio_steps=30,
    )
    t_tts_4 = time.time() - t0
    dur_tts_4 = len(res_tts_4.audio) / res_tts_4.sample_rate
    print(f"✓ TTS 4-bit Latency: {t_tts_4:.2f}s | Output: {dur_tts_4:.2f}s | RTF: {t_tts_4 / dur_tts_4:.4f}")
    assert dur_tts_4 > 0, "TTS output audio empty"

    print("=" * 75)
    print("ALL PRE-QUANTIZED DISK MODELS LOADED AND VERIFIED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    test_quantized_models_from_disk()
