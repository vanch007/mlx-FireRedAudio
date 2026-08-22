"""RedAE VAE Encoder and Decoder with ISTFT Vocoder Head in MLX."""

import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .backbone import RMSNorm, SwiGLU, apply_rope


class Qwen3SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def _build_rope(self, seq_len: int, offset: int = 0) -> mx.array:
        dim = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return freqs[None, :, None, :]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T, _ = x.shape
        offset = cache[0].shape[1] if cache is not None else 0

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        freqs = self._build_rope(T, offset)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=1)
            v = mx.concatenate([cache[1], v], axis=1)
        new_cache = (k, v)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        if mask is not None:
            scores = scores + mask
        attention = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attention @ v
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, T, -1)
        return self.o_proj(out), new_cache


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_size: int = 3584,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 128,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.self_attn = Qwen3SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Any]:
        h = self.input_layernorm(x)
        attn_out, new_cache = self.self_attn(h, mask=mask, cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


class Qwen3AudioBackbone(nn.Module):
    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_size: int = 3584,
        num_hidden_layers: int = 18,
        num_heads: int = 14,
        num_kv_heads: int = 2,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(151936, hidden_size)
        self.layers = [
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = RMSNorm(hidden_size, eps=1e-6)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        if mask is None and x.shape[1] > 1:
            # Qwen3Model is a causal decoder even when used inside RedAE.
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(x.dtype)
        for layer in self.layers:
            x, _ = layer(x, mask=mask)
        x = self.norm(x)
        return x


class ISTFTHeadMLX(nn.Module):
    def __init__(self, dim: int = 896, n_fft: int = 1920, hop_length: int = 480):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2, bias=True)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # x: (B, T, H) -> (B, T, n_fft + 2)
        pred = self.out(x)
        half_bins = self.n_fft // 2 + 1
        log_mag = pred[..., :half_bins]
        phase = pred[..., half_bins:]

        mag = mx.clip(mx.exp(log_mag), a_min=1e-6, a_max=100.0)
        real = mag * mx.cos(phase)
        imag = mag * mx.sin(phase)
        return real, imag

    def decode_waveform(self, real: mx.array, imag: mx.array) -> mx.array:
        """MLX-native inverse STFT with periodic Hann overlap-add."""
        B, T, _ = real.shape
        n_fft = self.n_fft
        hop = self.hop_length
        win = mx.hanning(n_fft + 1)[:-1].astype(mx.float32)
        frames = mx.fft.irfft(real + 1j * imag, n=n_fft, axis=-1) * win

        audios = []
        output_len = (T - 1) * hop + n_fft
        positions = (
            mx.arange(T, dtype=mx.int32)[:, None] * hop
            + mx.arange(n_fft, dtype=mx.int32)[None, :]
        )
        envelope_updates = mx.broadcast_to(mx.square(win), (T, n_fft))
        for b in range(B):
            y = mx.zeros((output_len,), dtype=mx.float32).at[positions].add(frames[b])
            window_sum = mx.zeros((output_len,), dtype=mx.float32).at[positions].add(
                envelope_updates
            )
            y = mx.where(window_sum > 1e-11, y / mx.maximum(window_sum, 1e-11), 0.0)

            pad = (n_fft - hop) // 2
            if output_len > 2 * pad:
                y = y[pad:-pad]
            audios.append(y)

        return mx.stack(audios, axis=0)


class RedAEDecoderMLX(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        upsample_rate: int = 2,
        hidden_size: int = 896,
        intermediate_size: int = 3584,
        num_layers: int = 18,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        audio_patch_size: int = 480,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, upsample_rate * hidden_size, bias=True)
        self.upsample_rate = upsample_rate
        self.hidden_size = hidden_size

        self.qwen3 = Qwen3AudioBackbone(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        self.istft_head = ISTFTHeadMLX(
            dim=hidden_size,
            n_fft=audio_patch_size * 4,
            hop_length=audio_patch_size,
        )

    def __call__(self, latents: mx.array) -> np.ndarray:
        B, T, _ = latents.shape
        h = self.in_proj(latents).reshape(B, T * self.upsample_rate, self.hidden_size)

        h = self.qwen3(h)
        real, imag = self.istft_head(h)
        audio = self.istft_head.decode_waveform(real, imag)
        mx.eval(audio)
        return np.array(audio)


class RedAEDownsample(nn.Module):
    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_size: int = 3584,
        num_layers: int = 4,
        num_heads: int = 14,
        num_kv_heads: int = 2,
    ):
        super().__init__()
        self.cls_tok = mx.zeros((1, 1, hidden_size))
        self.qwen3 = Qwen3AudioBackbone(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        x_reshaped = x.reshape(-1, 2, C)
        cls_tokens = mx.repeat(self.cls_tok, x_reshaped.shape[0], axis=0)
        x_cat = mx.concatenate([x_reshaped, cls_tokens], axis=1)
        out = self.qwen3(x_cat)
        cls_out = out[:, -1, :]
        return cls_out.reshape(B, T // 2, C)


class RedAEEncoderMLX(nn.Module):
    def __init__(
        self,
        audio_patch_size: int = 480,
        hidden_size: int = 896,
        intermediate_size: int = 3584,
        num_layers: int = 18,
        out_dim: int = 64,
    ):
        super().__init__()
        self.audio_patch_size = audio_patch_size
        self.in_proj = nn.Sequential(
            nn.Linear(audio_patch_size, hidden_size, bias=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.qwen3 = Qwen3AudioBackbone(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
        )
        self.downsample = RedAEDownsample(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=4,
        )
        self.out_proj = nn.Linear(hidden_size, out_dim, bias=True)

    def encode(self, audio: np.ndarray) -> mx.array:
        n_patches = len(audio) // self.audio_patch_size
        if n_patches == 0:
            return mx.zeros((1, 0, 64))

        patches = audio[: n_patches * self.audio_patch_size].reshape(1, n_patches, self.audio_patch_size)
        x = mx.array(patches, dtype=mx.float32)

        # The pretrained RedAE encoder uses two consecutive linear projections
        # with no activation between them (matching torch.nn.Sequential in the
        # upstream implementation).
        h = self.in_proj.layers[0](x)
        h = self.in_proj.layers[1](h)
        h = self.qwen3(h)
        h_down = self.downsample(h)
        latents = self.out_proj(h_down)
        return latents
