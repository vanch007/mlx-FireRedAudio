#!/usr/bin/env python3
"""Content-level regression for the two FireRedAudio synthesis entry points."""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from fireredaudio_mlx import FireRedAudioInference


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w]", "", text, flags=re.UNICODE).lower()


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def content_error_rate(expected: str, actual: str) -> float:
    expected = normalize_text(expected)
    actual = normalize_text(actual)
    return edit_distance(expected, actual) / max(1, len(expected))


def estimate_snr_db(audio: np.ndarray, sr: int = 24000) -> float:
    frame_len = int(0.02 * sr)
    frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    if not frames:
        return 0.0
    energies = [np.mean(f.astype(np.float64) ** 2) for f in frames]
    sorted_e = np.sort(energies)
    noise_floor = np.mean(sorted_e[: max(1, len(sorted_e) // 10)]) + 1e-12
    signal_peak = np.mean(sorted_e[-max(1, len(sorted_e) // 10) :]) + 1e-12
    return float(10.0 * np.log10(signal_peak / noise_floor))


def validate_audio(audio: np.ndarray, sample_rate: int, label: str) -> float:
    duration = len(audio) / sample_rate
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) <= 0 or not 0.5 <= duration <= 30:
        raise AssertionError(f"{label} produced invalid audio: duration={duration:.2f}s")
    return duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/FireRedAudio-8bit")
    parser.add_argument("--max-cer", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    engine = FireRedAudioInference(args.model)
    results = {}

    tts_target = "安徽淮南秦师傅发现，停在小区的爱车右前驾驶窗玻璃被砸。"
    mx.random.seed(42)
    tts = engine.tts(
        prompt_text="同时，他强调微调要科学有序。",
        prompt_audio="assets/examples/tts_zh_prompt.wav",
        target_text=tts_target,
        n_timesteps=10,
        min_new_audio_steps=6,
        max_new_audio_steps=80,
    )
    tts_path = output_dir / "regression_tts_content.wav"
    sf.write(tts_path, tts.audio, tts.sample_rate, subtype="PCM_16")
    tts_duration = validate_audio(tts.audio, tts.sample_rate, "tts")
    tts_snr = estimate_snr_db(tts.audio, tts.sample_rate)
    tts_transcript = engine.understand(
        str(tts_path), task="asr", max_new_tokens=120, num_beams=1
    ).answer
    tts_cer = content_error_rate(tts_target, tts_transcript)
    if tts_cer > args.max_cer:
        raise AssertionError(f"TTS content regression: CER={tts_cer:.4f}, transcript={tts_transcript!r}")
    results["tts"] = {
        "target": tts_target,
        "transcript": tts_transcript,
        "cer": round(tts_cer, 4),
        "duration_s": round(tts_duration, 2),
        "snr_db": round(tts_snr, 2),
        "output": str(tts_path),
    }

    voice_target = "是我请他来的，可他什么也不知道，他来只是想打听一下，你们厂是不是有旧锅炉？"
    mx.random.seed(42)
    voice = engine.voice_design(
        instruction="以女性高音区的清亮音色,表现出青年阶段的特质,音量略强,语速适中稍快,语调带有解释意味和急切的情感流露,确保语音流畅自然。",
        text=voice_target,
        n_timesteps=10,
        min_new_audio_steps=6,
        max_new_audio_steps=80,
        max_new_text_tokens=512,
    )
    voice_path = output_dir / "regression_voice_design_content.wav"
    sf.write(voice_path, voice.audio, voice.sample_rate, subtype="PCM_16")
    voice_duration = validate_audio(voice.audio, voice.sample_rate, "voice_design")
    voice_snr = estimate_snr_db(voice.audio, voice.sample_rate)
    voice_transcript = engine.understand(
        str(voice_path), task="asr", max_new_tokens=120, num_beams=1
    ).answer
    voice_cer = content_error_rate(voice_target, voice_transcript)
    if voice_cer > args.max_cer:
        raise AssertionError(
            f"Voice Design content regression: CER={voice_cer:.4f}, transcript={voice_transcript!r}"
        )
    results["voice_design"] = {
        "target": voice_target,
        "transcript": voice_transcript,
        "cer": round(voice_cer, 4),
        "duration_s": round(voice_duration, 2),
        "snr_db": round(voice_snr, 2),
        "timbre": voice.text,
        "output": str(voice_path),
    }

    print(json.dumps({"status": "pass", **results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
