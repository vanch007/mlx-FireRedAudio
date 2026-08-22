"""Qwen3.5 LLM Backbone with Gated Delta Net and Full Attention in MLX."""

import math
from typing import Optional, Tuple, List, Dict, Any, Union
import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.gated_delta import gated_delta_update as _metal_gated_delta_update
except ImportError:  # pragma: no cover - retained for minimal installations
    _metal_gated_delta_update = None


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        return (x * mx.rsqrt(variance + self.eps)).astype(x.dtype) * self.weight


class GatedRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array, z: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        normed = (x * mx.rsqrt(variance + self.eps)).astype(x.dtype) * self.weight
        return normed * nn.silu(z)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.sum(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + eps).astype(x.dtype)

def _gated_delta_scan(q, k, v, g_exp, beta, init_state):
    state = init_state
    outputs = []
    T = q.shape[2]
    for t in range(T):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        g_t = g_exp[:, :, t, None, None]
        b_t = beta[:, :, t, None]

        state = state * g_t
        kv_mem = mx.sum(state * k_t[:, :, :, None], axis=-2)
        delta = (v_t - kv_mem) * b_t
        state = state + k_t[:, :, :, None] * delta[:, :, None, :]
        out_t = mx.sum(state * q_t[:, :, :, None], axis=-2)
        outputs.append(out_t)

    return mx.stack(outputs, axis=2), state

_compiled_gated_delta_scan = mx.compile(_gated_delta_scan)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    d = freqs.shape[-1]
    x_rot, x_pass = x[..., :d], x[..., d:]
    d_half = d // 2
    x1, x2 = x_rot[..., :d_half], x_rot[..., d_half:]
    x_rotated = mx.concatenate([-x2, x1], axis=-1)
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    x_out = (x_rot * cos) + (x_rotated * sin)
    if x_pass.shape[-1] > 0:
        x_out = mx.concatenate([x_out, x_pass], axis=-1)
    return x_out.astype(x.dtype)


class Qwen3_5Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 256,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.rope_dim = int(head_dim * partial_rotary_factor)
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def _build_rope(self, seq_len: int, offset: int = 0) -> mx.array:
        dim = self.rope_dim
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

        q_gate = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim * 2)
        q = q_gate[..., : self.head_dim]
        gate = q_gate[..., self.head_dim :].reshape(B, T, -1)

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
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = mx.transpose(out, (0, 2, 1, 3))

        out = out.reshape(B, T, -1)
        out = out * mx.sigmoid(gate)
        return self.o_proj(out), new_cache


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_size: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size

        self.in_proj_qkv = nn.Linear(hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.key_dim * 2 + self.value_dim,
            out_channels=self.key_dim * 2 + self.value_dim,
            kernel_size=conv_kernel_size,
            padding=0,
            groups=self.key_dim * 2 + self.value_dim,
            bias=False,
        )
        self.dt_bias = mx.zeros((num_v_heads,))
        self.A_log = mx.zeros((num_v_heads,))

        self.norm = GatedRMSNorm(head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _apply_causal_conv(self, x: mx.array, conv_state: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        B, T, C = x.shape
        K = self.conv_kernel_size

        if conv_state is not None:
            x_padded = mx.concatenate([conv_state, x], axis=1)
        else:
            x_padded = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])

        new_conv_state = x_padded[:, -(K - 1) :, :]
        y = self.conv1d(x_padded)
        y = nn.silu(y[:, :T, :])
        return y, new_conv_state

    def __call__(
        self,
        x: mx.array,
        recurrent_state: Optional[mx.array] = None,
        conv_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        B, T, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x)
        mixed_qkv, new_conv_state = self._apply_causal_conv(mixed_qkv, conv_state)

        z = self.in_proj_z(x).reshape(B, T, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        q_raw = mixed_qkv[:, :, : self.key_dim].reshape(B, T, self.num_k_heads, self.head_k_dim)
        k_raw = mixed_qkv[:, :, self.key_dim : 2 * self.key_dim].reshape(B, T, self.num_k_heads, self.head_k_dim)
        v_raw = mixed_qkv[:, :, 2 * self.key_dim :].reshape(B, T, self.num_v_heads, self.head_v_dim)

        inv_scale = self.head_k_dim ** -0.5
        q_norm = (inv_scale**2) * mx.fast.rms_norm(q_raw, None, 1e-6)
        k_norm = inv_scale * mx.fast.rms_norm(k_raw, None, 1e-6)

        if _metal_gated_delta_update is not None:
            core_attn_out, state = _metal_gated_delta_update(
                q_norm,
                k_norm,
                v_raw,
                a,
                b,
                self.A_log,
                self.dt_bias,
                recurrent_state,
                use_kernel=True,
            )
        else:
            beta = mx.sigmoid(b)
            g = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(
                a.astype(mx.float32) + self.dt_bias
            )
            if self.num_v_heads != self.num_k_heads:
                rep = self.num_v_heads // self.num_k_heads
                q_norm = mx.repeat(q_norm, rep, axis=2)
                k_norm = mx.repeat(k_norm, rep, axis=2)
            state = recurrent_state
            if state is None:
                state = mx.zeros(
                    (B, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    dtype=mx.float32,
                )
            core_attn_out, state = _compiled_gated_delta_scan(
                mx.transpose(q_norm, (0, 2, 1, 3)),
                mx.transpose(k_norm, (0, 2, 1, 3)),
                mx.transpose(v_raw, (0, 2, 1, 3)),
                mx.transpose(mx.exp(g), (0, 2, 1)),
                mx.transpose(beta, (0, 2, 1)),
                state,
            )
            core_attn_out = mx.transpose(core_attn_out, (0, 2, 1, 3))
        normed = self.norm(core_attn_out, z).reshape(B, T, -1)
        out = self.out_proj(normed)
        return out, state, new_conv_state


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, layer_type: str = "linear_attention", hidden_size: int = 4096, intermediate_size: int = 12288):
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(hidden_size=hidden_size)
        else:
            self.self_attn = Qwen3_5Attention(hidden_size=hidden_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Any]:
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            rec_state = cache[0] if cache is not None else None
            conv_state = cache[1] if cache is not None else None
            attn_out, new_rec, new_conv = self.linear_attn(h, rec_state, conv_state)
            new_cache = (new_rec, new_conv)
        else:
            attn_out, new_cache = self.self_attn(h, mask=mask, cache=cache)

        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


class Qwen3_5LanguageModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        layer_types = config.get("layer_types", ["linear_attention"] * 32)
        self.layers = [
            Qwen3_5DecoderLayer(
                layer_type=lt,
                hidden_size=config["hidden_size"],
                intermediate_size=config["intermediate_size"]
            )
            for lt in layer_types
        ]
        self.norm = RMSNorm(config["hidden_size"], eps=1e-6)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        caches: Optional[List[Any]] = None,
    ) -> Tuple[mx.array, List[Any]]:
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        T = x.shape[1]
        if mask is None and T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        new_caches = []
        for i, layer in enumerate(self.layers):
            c = caches[i] if caches is not None else None
            x, nc = layer(x, mask=mask, cache=c)
            new_caches.append(nc)

        x = self.norm(x)
        return x, new_caches


class FireRedAudioBackbone(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.language_model = Qwen3_5LanguageModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        caches: Optional[List[Any]] = None,
    ) -> Tuple[mx.array, List[Any]]:
        hidden, new_caches = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            mask=mask,
            caches=caches,
        )
        return hidden, new_caches
