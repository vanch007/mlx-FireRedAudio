"""Audio loading and resampling utilities for MLX."""

import math
import subprocess
import numpy as np
import soundfile as sf

UNDERSTAND_SAMPLE_RATE = 16000
GENERATION_SAMPLE_RATE = 24000
VAE_DOWNSAMPLE_RATE = 960
PATCH_ENCODER_DOWNSAMPLE_RATE = VAE_DOWNSAMPLE_RATE * 4  # 3840 samples


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D audio numpy array to target sample rate."""
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def read_audio(path: str, target_sample_rate: int) -> np.ndarray:
    """Load audio file as 1-D float32 mono array resampled to target_sample_rate."""
    try:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if sr != target_sample_rate:
            data = resample_audio(data, sr, target_sample_rate)
        return data.astype(np.float32)
    except Exception:
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
            "-f", "s16le", "-ac", "1", "-ar", str(target_sample_rate), "-"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return data


def pad_to_multiple_of(audio: np.ndarray, multiple: int = PATCH_ENCODER_DOWNSAMPLE_RATE) -> np.ndarray:
    """Pad audio to a multiple of patch downsample rate."""
    length = audio.shape[-1]
    target_samples = math.ceil(length / multiple) * multiple
    pad_len = target_samples - length
    if pad_len > 0:
        audio = np.pad(audio, (0, pad_len), mode="constant")
    return audio
