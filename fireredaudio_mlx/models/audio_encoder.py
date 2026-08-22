"""Audio Encoder for speech understanding in MLX."""

import math
from typing import Dict, Any, List, Optional
import mlx.core as mx
import mlx.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int = 1280, n_heads: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        T, D = x.shape
        q = self.q_proj(x).reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)  # (H, T, D_h)
        k = self.k_proj(x).reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
        v = self.v_proj(x).reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)

        if T >= 1024:
            out = mx.fast.scaled_dot_product_attention(
                q[None], k[None], v[None], scale=self.scale, mask=mask
            )[0]
        else:
            scores = (q @ k.transpose(0, 2, 1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attention = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = attention @ v
        out = out.transpose(1, 0, 2).reshape(T, D)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 1280, n_heads: int = 20, ffn_dim: int = 5120):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # Self attention with residual
        h = self.self_attn_layer_norm(x)
        h = self.self_attn(h)
        x = x + h

        # FFN with residual
        h = self.final_layer_norm(x)
        h = self.fc2(nn.gelu(self.fc1(h)))
        x = x + h
        return x


class AudioEncoderAdapter(nn.Module):
    def __init__(self, d_model: int = 1280, output_dim: int = 4096):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, output_dim, bias=True)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (T, D) -> (1, T, D) for Conv1d
        x = x[None, :, :]
        x = self.conv3(x)
        x = self.conv4(x)
        x = x[0]  # (T_out, D)
        x = self.layer_norm(x)
        x = self.linear2(nn.gelu(self.linear1(x)))
        return x


class FireRedAudioEncoderMLX(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        d_model = config.get("d_model", 1280)
        output_dim = config.get("output_dim", 4096)
        n_layers = config.get("encoder_layers", 32)
        n_heads = config.get("encoder_attention_heads", 20)
        ffn_dim = config.get("encoder_ffn_dim", 5120)
        max_pos = config.get("max_source_positions", 1500)
        self.n_window = config.get("n_window", 1500)

        # Input 1D Convolutions
        self.conv1 = nn.Conv1d(128, d_model, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=True)

        self.max_source_positions = max_pos
        self.d_model = d_model
        self.layers = [AudioEncoderLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        self.adapter = AudioEncoderAdapter(d_model, output_dim)

    def _sinusoidal_positional_embedding(self, length: int) -> mx.array:
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        freqs = mx.exp(mx.arange(0, half_dim, dtype=mx.float32) * -emb)
        t = mx.arange(0, length, dtype=mx.float32)
        args = t[:, None] * freqs[None, :]
        sin_emb = mx.sin(args)
        cos_emb = mx.cos(args)
        return mx.concatenate([sin_emb, cos_emb], axis=-1)

    def _encode_chunk(self, mel: mx.array) -> mx.array:
        """Encode at most 2*n_window mel frames with positions reset per chunk."""
        x = mel.transpose(1, 0)[None, :, :]
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))[0]
        x = x + self._sinusoidal_positional_embedding(x.shape[0])
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, mel: mx.array) -> mx.array:
        # mel: (128, T) or (1, 128, T)
        if mel.ndim == 3:
            mel = mel[0]
        chunk_size = self.n_window * 2
        encoded_chunks = [
            self._encode_chunk(mel[:, start : start + chunk_size])
            for start in range(0, mel.shape[1], chunk_size)
        ]
        x = mx.concatenate(encoded_chunks, axis=0)
        # Upstream concatenates valid chunk outputs before the stride-4 adapter.
        out = self.adapter(x)
        return out
