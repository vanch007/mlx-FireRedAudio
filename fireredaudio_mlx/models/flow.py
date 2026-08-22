"""Flow matching DiT and Patch Encoder for generation in MLX."""

import math
from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from .backbone import RMSNorm


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

def rotate_half_pairs(x: mx.array) -> mx.array:
    B, H, T, D = x.shape
    x_pair = x.reshape(B, H, T, D // 2, 2)
    x1 = x_pair[..., 0]
    x2 = x_pair[..., 1]
    return mx.stack([-x2, x1], axis=-1).reshape(B, H, T, D)


def apply_rope_flow(x: mx.array, seq_len: int) -> mx.array:
    dim = x.shape[-1]
    inv_freq = 1.0 / (10000.0 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(0, seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)
    freqs = mx.stack([freqs, freqs], axis=-1).reshape(seq_len, dim)
    cos = mx.cos(freqs)[None, None, :, :]
    sin = mx.sin(freqs)[None, None, :, :]
    return (x * cos) + (rotate_half_pairs(x) * sin)


class SelfAttention(nn.Module):
    def __init__(self, dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = [nn.Linear(dim, dim, bias=True)]

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        q = self.to_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.to_k(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = apply_rope_flow(q, T)
        k = apply_rope_flow(k, T)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attention = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (attention @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.to_out[0](out)


class PatchEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Sequential(nn.Linear(hidden_size, int(hidden_size * mlp_ratio)), nn.GELU()),
            nn.Identity(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        h = self.mlp.layers[0].layers[1](self.mlp.layers[0].layers[0](h))
        h = self.mlp.layers[2](h)
        x = x + h
        return x


class RedPatchEncoderMLX(nn.Module):
    def __init__(
        self,
        vae_dim: int = 64,
        patch_size: int = 4,
        hidden_size: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        out_dim: int = 4096,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.vae_dim = vae_dim

        in_dim = vae_dim
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.cls_tok = mx.zeros((1, 1, hidden_size))
        self.blocks = [PatchEncoderBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        self.out_proj = nn.Sequential(
            RMSNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, out_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        h = self.in_proj.layers[0](x)
        h = self.in_proj.layers[2](nn.gelu(h))

        h = h.reshape(-1, self.patch_size, 1024)
        cls_tokens = mx.repeat(self.cls_tok, h.shape[0], axis=0)
        h = mx.concatenate([cls_tokens, h], axis=1)

        for block in self.blocks:
            h = block(h)

        h_out = self.out_proj.layers[0](h[:, 0:1, :])
        out = self.out_proj.layers[1](h_out)
        out = out.reshape(B, -1, 4096)
        return out


class TimestepEmbedderMLX(nn.Module):
    def __init__(self, dim: int = 1024, freq_embed_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.freq_embed_dim = freq_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoid(self, t: mx.array) -> mx.array:
        half_dim = self.freq_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        freqs = mx.exp(mx.arange(0, half_dim, dtype=mx.float32) * -emb)
        args = t[:, None] * freqs[None, :] * 1000.0
        return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)

    def __call__(self, t: mx.array) -> mx.array:
        emb = self._sinusoid(t)
        h = nn.silu(self.time_mlp.layers[0](emb))
        return self.time_mlp.layers[2](h)


class ConvBlockMLX(nn.Module):
    def __init__(self, dim: int = 1024, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.block.layers[0](x)
        h = nn.mish(h)
        h = self.block.layers[2](h)
        return h


class DiTBlockMLX(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.conv = ConvBlockMLX(hidden_size)
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Sequential(nn.Linear(hidden_size, int(hidden_size * mlp_ratio)), nn.GELU()),
            nn.Identity(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mods = self.adaLN_modulation.layers[1](nn.silu(c))
        mods = mx.split(mods, 9, axis=-1)
        # Match the pretrained PyTorch chunk order exactly:
        # attention, MLP, then convolution.
        shift_msa, scale_msa, gate_msa = mods[0], mods[1], mods[2]
        shift_mlp, scale_mlp, gate_mlp = mods[3], mods[4], mods[5]
        shift_conv, scale_conv, gate_conv = mods[6], mods[7], mods[8]

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa[:, None, :] * self.attn(h)

        h = modulate(self.norm2(x), shift_conv, scale_conv)
        x = x + gate_conv[:, None, :] * self.conv(h)

        h = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_h = self.mlp.layers[0].layers[1](self.mlp.layers[0].layers[0](h))
        mlp_h = self.mlp.layers[2](mlp_h)
        x = x + gate_mlp[:, None, :] * mlp_h
        return x


class FinalLayerMLX(nn.Module):
    def __init__(self, hidden_size: int = 1024, out_channels: int = 64):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mods = self.adaLN_modulation.layers[1](nn.silu(c))
        shift, scale = mx.split(mods, 2, axis=-1)
        # Upstream FinalLayer uses affine-free LayerNorm, not RMSNorm.
        x32 = x.astype(mx.float32)
        mean = mx.mean(x32, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
        norm_x = ((x32 - mean) * mx.rsqrt(variance + 1e-6)).astype(x.dtype)
        norm_x = norm_x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.linear(norm_x)


class RedDiTMLX(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config.get("hidden_size", 1024)
        vae_channels = config.get("vae_channels", 64)
        backbone_dim = config.get("backbone_hidden_size", 4096)
        num_heads = config.get("num_heads", 16)
        depth = config.get("depth", 11)
        self.patch_size = config.get("patch_size", 4)
        self.vae_channels = vae_channels

        self.t_embedder = TimestepEmbedderMLX(hidden_size, 256)
        self.in_proj = nn.Linear(vae_channels + hidden_size, hidden_size, bias=True)
        self.backbone_input_proj = nn.Linear(backbone_dim, hidden_size, bias=True)

        self.blocks = [DiTBlockMLX(hidden_size, num_heads) for _ in range(depth)]
        self.final_layer = FinalLayerMLX(hidden_size, vae_channels)
        self._compiled_forward = mx.compile(self._forward_estimator)

    def _forward_estimator(self, x: mx.array, t_emb: mx.array) -> mx.array:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        v = self.final_layer(h, t_emb)
        return v

    def generate(
        self,
        backbone_output: mx.array,
        history_vae_latents: Optional[mx.array] = None,
        n_timesteps: int = 10,
        inference_cfg: float = 1.5,
        solver: str = "euler",
    ) -> mx.array:
        b = backbone_output.shape[0]
        model_dtype = backbone_output.dtype
        t_span = mx.linspace(0.0, 1.0, n_timesteps + 1).astype(model_dtype)
        t_span = 1.0 - mx.cos(t_span * 0.5 * math.pi)

        backbone_pad_len = 3 - backbone_output.shape[1]
        if backbone_pad_len > 0:
            backbone_output = mx.pad(backbone_output, [(0, 0), (backbone_pad_len, 0), (0, 0)])

        if history_vae_latents is None:
            history_vae_latents = mx.zeros(
                (b, 8, self.vae_channels), dtype=model_dtype
            )
        elif history_vae_latents.shape[1] < 8:
            pad_len = 8 - history_vae_latents.shape[1]
            history_vae_latents = mx.pad(history_vae_latents, [(0, 0), (pad_len, 0), (0, 0)])

        dit_backbone_cond = mx.repeat(backbone_output[:, -3:], self.patch_size, axis=1)
        dit_cond = self.backbone_input_proj(dit_backbone_cond)

        noise = mx.random.normal(
            (b, self.patch_size, self.vae_channels)
        ).astype(history_vae_latents.dtype)
        history_slice = history_vae_latents[:, -8:]

        def get_v_guided(cur_x: mx.array, t_val: mx.array) -> mx.array:
            x_full = mx.concatenate([history_slice, cur_x], axis=1)
            xt_in = mx.concatenate([
                mx.concatenate([x_full, dit_cond], axis=-1),
                mx.concatenate([x_full, mx.zeros_like(dit_cond)], axis=-1),
            ], axis=0)
            t_in = mx.repeat(t_val, b * 2, axis=0)
            t_emb = self.t_embedder(t_in)
            vt = self._compiled_forward(xt_in, t_emb)[:, 8:]
            vt_cond, vt_cfg = vt[:b], vt[b:]
            return (1.0 + inference_cfg) * vt_cond - inference_cfg * vt_cfg

        cur_patch = noise
        for step in range(n_timesteps):
            t = t_span[step : step + 1]
            dt = (t_span[step + 1] - t_span[step]).item()
            if solver == "midpoint":
                t_mid = (t + t_span[step + 1 : step + 2]) * 0.5
                v1 = get_v_guided(cur_patch, t)
                x_mid = cur_patch + (0.5 * dt) * v1
                v2 = get_v_guided(x_mid, t_mid)
                cur_patch = cur_patch + dt * v2
            elif solver == "heun":
                t_next = t_span[step + 1 : step + 2]
                v1 = get_v_guided(cur_patch, t)
                x_pred = cur_patch + dt * v1
                v2 = get_v_guided(x_pred, t_next)
                cur_patch = cur_patch + (0.5 * dt) * (v1 + v2)
            else:
                v = get_v_guided(cur_patch, t)
                cur_patch = cur_patch + dt * v

        return cur_patch
