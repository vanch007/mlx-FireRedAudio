"""Unified FireRedAudio Model in MLX."""

import math
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from .backbone import FireRedAudioBackbone
from .audio_encoder import FireRedAudioEncoderMLX
from .flow import RedPatchEncoderMLX, RedDiTMLX
from .redae import RedAEEncoderMLX, RedAEDecoderMLX

logger = logging.getLogger(__name__)


def sample_top_k_top_p(
    logits: mx.array,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
) -> int:
    """Sample one token using the upstream FireRedAudio text policy."""
    scores = logits.astype(mx.float32) / temperature
    k = min(top_k, scores.shape[-1])
    top_ids = mx.argpartition(scores, -k, axis=-1)[:, -k:]
    top_scores = mx.take_along_axis(scores, top_ids, axis=-1)
    order = mx.argsort(top_scores, axis=-1)[:, ::-1]
    top_ids = mx.take_along_axis(top_ids, order, axis=-1)
    top_scores = mx.take_along_axis(top_scores, order, axis=-1)

    probs = mx.softmax(top_scores, axis=-1)
    cumulative = mx.cumsum(probs, axis=-1)
    # Keep the first token that crosses top_p, matching common nucleus filters.
    remove = (cumulative - probs) >= top_p
    filtered = mx.where(remove, mx.array(-float("inf")), top_scores)
    sampled_rank = int(mx.random.categorical(filtered)[0].item())
    return int(top_ids[0, sampled_rank].item())


def sanitize_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Map safetensor keys and transpose conv weights for MLX."""
    sanitized = {}
    # Qwen3.5 stores the backbone RMSNorm parameters as zero-centered deltas
    # and applies ``1 + weight`` in the PyTorch implementation.  MLX RMSNorm
    # expects the effective scale, so these parameters must be shifted while
    # loading.  The gated linear-attention norm and the Qwen3 modules used by
    # RedAE use ordinary RMSNorm weights and must not be shifted.
    qwen35_shifted_norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".language_model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for k, v in weights.items():
        # Ignore buffers
        if "istft.window" in k:
            continue

        # 1. backbone_llm.model.language_model -> backbone_llm.language_model
        if k.startswith("backbone_llm.model.language_model."):
            k = "backbone_llm.language_model." + k[len("backbone_llm.model.language_model.") :]

        if k.startswith("backbone_llm.language_model.") and k.endswith(
            qwen35_shifted_norm_suffixes
        ):
            v = v + 1.0

        # 2. Sequential indexing mappings
        k = re.sub(r"\.time_mlp\.(\d+)", r".time_mlp.layers.\1", k)
        k = re.sub(r"\.adaLN_modulation\.(\d+)", r".adaLN_modulation.layers.\1", k)
        k = re.sub(r"\.conv\.block\.(\d+)", r".conv.block.layers.\1", k)
        k = re.sub(r"\.mlp\.ff\.0\.0", r".mlp.layers.0.layers.0", k)
        k = re.sub(r"\.mlp\.ff\.2", r".mlp.layers.2", k)
        k = re.sub(r"patch_encoder\.in_proj\.(\d+)", r"patch_encoder.in_proj.layers.\1", k)
        k = k.replace("patch_encoder.out_proj.norm_final", "patch_encoder.out_proj.layers.0")
        k = k.replace("patch_encoder.out_proj.linear", "patch_encoder.out_proj.layers.1")
        k = re.sub(r"red_vae\.in_proj\.(\d+)", r"red_vae.in_proj.layers.\1", k)

        # 3. Transpose 1D Conv weights: PyTorch (C_out, C_in, K) -> MLX (C_out, K, C_in)
        if ("conv" in k or "conv1d" in k) and k.endswith(".weight") and v.ndim == 3:
            v = v.transpose(0, 2, 1)

        sanitized[k] = v
    return sanitized


def apply_native_weight_corrections(
    weights: Dict[str, mx.array], config: Dict[str, Any]
) -> Dict[str, mx.array]:
    """Apply explicit, auditable corrections for previously exported MLX weights."""
    legacy_add_one_keys = (
        "backbone_llm.language_model.layers.7.self_attn.q_norm.weight",
        "backbone_llm.language_model.layers.11.self_attn.q_norm.weight",
        "backbone_llm.language_model.layers.11.self_attn.k_norm.weight",
        "backbone_llm.language_model.layers.15.self_attn.q_norm.weight",
        "backbone_llm.language_model.layers.15.self_attn.k_norm.weight",
        "backbone_llm.language_model.layers.19.self_attn.q_norm.weight",
        "backbone_llm.language_model.layers.19.self_attn.k_norm.weight",
        "backbone_llm.language_model.layers.23.self_attn.q_norm.weight",
        "backbone_llm.language_model.layers.23.self_attn.k_norm.weight",
        "backbone_llm.language_model.norm.weight",
    )
    explicit_add_one_keys = set(
        config.get("mlx_native_weight_corrections", {}).get("add_one", [])
    )
    add_one_keys = set(explicit_add_one_keys)
    # Quantized artifacts exported before the effective-scale marker used a
    # BF16 mean heuristic and deterministically left these norms unshifted.
    if (
        config.get("quantization")
        and config.get("qwen35_norm_format") != "effective_scale_v2"
    ):
        add_one_keys.update(key for key in legacy_add_one_keys if key in weights)
    if not add_one_keys:
        return weights
    corrected = dict(weights)
    for key in sorted(add_one_keys):
        if key not in corrected and key in explicit_add_one_keys:
            raise KeyError(f"native weight correction key is missing: {key}")
        if key not in corrected:
            continue
        corrected[key] = corrected[key] + 1.0
    return corrected


class FireRedAudioModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.sosp_idx = config.get("sosp_idx", 248077)
        self.eosp_idx = config.get("eosp_idx", 248078)
        self.audio_special_token_id = config.get("audio_special_token_id", 248091)
        self.audio_special_no_latent_id = config.get("audio_special_no_latent_id", 248092)

        # 1. Backbone LLM
        self.backbone_llm = FireRedAudioBackbone(config.get("backbone_config", {}))

        # 2. Audio Encoder (Understanding)
        self.audio_encoder = FireRedAudioEncoderMLX(config.get("audio_encoder_config", {}))

        # 3. Patch Encoder (Generation)
        self.patch_encoder = RedPatchEncoderMLX()

        # 4. Flow DiT (Generation Denoising Head)
        self.dit = RedDiTMLX(config.get("dit_config", {}))

        # 5. RedAE Encoder (Generation)
        self.red_vae = RedAEEncoderMLX()

        # 6. RedAE Decoder (Vocoder / Waveform Generator)
        self.vae_decoder = RedAEDecoderMLX()

    def load_weights(self, weights: Dict[str, mx.array]):
        """Load pretrained MLX safetensors dictionary into model parameters."""
        is_mlx_native = "quantization" in self.config or self.config.get("framework") == "mlx"
        sanitized = (
            apply_native_weight_corrections(weights, self.config)
            if is_mlx_native
            else sanitize_weights(weights)
        )
        unflattened = tree_unflatten(list(sanitized.items()))
        self.update(unflattened)
        logger.info("Successfully updated model weights from %d tensors (is_mlx_native=%s)", len(weights), is_mlx_native)

    def get_audio_features(self, input_features: mx.array) -> mx.array:
        """Extract continuous audio representations from 128-bin log-mel filterbank."""
        return self.audio_encoder(input_features)

    def generate(
        self,
        input_ids: mx.array,
        audio_features: Optional[Union[mx.array, List[mx.array]]] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.8,
        num_beams: int = 1,
        repetition_penalty: float = 1.15,
        eos_token_id: int = 248044,
    ) -> List[int]:
        """Autoregressive text generation for ASR and Audio Understanding."""
        ids_np = np.array(input_ids[0])
        embed_tokens = self.backbone_llm.language_model.embed_tokens
        text_embeds = embed_tokens(input_ids)[0]

        # Insert audio features at placeholder positions if present
        if isinstance(audio_features, list) and audio_features:
            feat = mx.concatenate(
                [self.get_audio_features(features) for features in audio_features],
                axis=0,
            )
        elif audio_features is not None and audio_features.shape[0] > 0:
            feat = self.get_audio_features(audio_features)
        else:
            feat = None

        if feat is not None and feat.shape[0] > 0:
            indices = np.where(ids_np == self.audio_special_token_id)[0]
            if len(indices) != feat.shape[0]:
                raise ValueError(
                    f"audio feature length ({feat.shape[0]}) does not match "
                    f"placeholder count ({len(indices)})"
                )
            if len(indices) > 0:
                # Replace each placeholder in place so text separating multiple
                # audio spans remains intact.
                pieces = []
                previous = 0
                for feature_idx, token_idx in enumerate(indices.tolist()):
                    pieces.append(text_embeds[previous:token_idx])
                    pieces.append(feat[feature_idx : feature_idx + 1])
                    previous = token_idx + 1
                pieces.append(text_embeds[previous:])
                embeds = mx.concatenate(pieces, axis=0)[None, :, :]
            else:
                embeds = text_embeds[None, :, :]
        else:
            embeds = text_embeds[None, :, :]

        # Prefill backbone
        h, caches = self.backbone_llm.language_model(inputs_embeds=embeds)
        current_h = h[:, -1:]

        if num_beams > 1:
            return self._beam_search(
                current_h=current_h,
                caches=caches,
                embed_tokens=embed_tokens,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                num_beams=num_beams,
            )

        generated_tokens = []
        for _ in range(max_new_tokens):
            logits = self.backbone_llm.lm_head(current_h[:, -1, :])
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                recent_tokens = list(set(generated_tokens[-64:]))
                if recent_tokens:
                    indices = mx.array([recent_tokens])
                    selected_logits = mx.take_along_axis(logits, indices, axis=-1)
                    penalized = mx.where(
                        selected_logits < 0,
                        selected_logits * repetition_penalty,
                        selected_logits / repetition_penalty,
                    )
                    logits = mx.put_along_axis(logits, indices, penalized, axis=-1)

            if temperature > 0.0:
                next_token = sample_top_k_top_p(logits, temperature, top_k, top_p)
            else:
                next_token = int(mx.argmax(logits, axis=-1)[0].item())

            generated_tokens.append(next_token)
            if next_token == eos_token_id:
                break

            next_embed = embed_tokens(mx.array([[next_token]]))
            current_h, caches = self.backbone_llm.language_model(
                inputs_embeds=next_embed,
                caches=caches,
            )

        return generated_tokens

    def _beam_search(
        self,
        current_h: mx.array,
        caches: List[Any],
        embed_tokens: nn.Embedding,
        max_new_tokens: int,
        eos_token_id: int,
        num_beams: int,
    ) -> List[int]:
        """Small batch-size-one beam search used by deterministic ASR."""
        beams = [([], 0.0, current_h, caches, False)]
        for _ in range(max_new_tokens):
            candidates = []
            for tokens, score, hidden, beam_cache, ended in beams:
                if ended:
                    candidates.append((tokens, score, hidden, beam_cache, True, None))
                    continue
                logits = self.backbone_llm.lm_head(hidden[:, -1, :]).astype(mx.float32)
                log_probs = nn.log_softmax(logits, axis=-1)
                mx.eval(log_probs)
                arr = np.array(log_probs[0], dtype=np.float32)
                top_indices = np.argpartition(arr, -num_beams)[-num_beams:]
                for token in top_indices:
                    token = int(token)
                    token_score = float(arr[token])
                    candidates.append((
                        tokens + [token],
                        score + token_score,
                        hidden,
                        beam_cache,
                        token == eos_token_id,
                        token,
                    ))

            candidates.sort(key=lambda item: item[1], reverse=True)
            selected = candidates[:num_beams]
            beams = []
            for tokens, score, hidden, beam_cache, ended, token in selected:
                if ended or token is None:
                    beams.append((tokens, score, hidden, beam_cache, ended))
                else:
                    next_embed = embed_tokens(mx.array([[token]]))
                    next_h, next_cache = self.backbone_llm.language_model(
                        inputs_embeds=next_embed,
                        caches=beam_cache,
                    )
                    beams.append((tokens, score, next_h, next_cache, False))
            if all(beam[4] for beam in beams):
                break

        best = max(
            beams,
            key=lambda item: item[1] / max(1, len(item[0])),
        )
        return best[0]

    def generate_tts(
        self,
        input_ids: mx.array,
        vae_audios: Optional[List[np.ndarray]] = None,
        vae_is_assistant: Optional[np.ndarray] = None,
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 6,
        max_new_text_tokens: int = 512,
        n_timesteps: int = 10,
        inference_cfg: float = 1.5,
        solver: str = "euler",
        eos_token_id: int = 248044,
    ) -> Tuple[List[int], np.ndarray, mx.array]:
        """Hybrid AR generation: text token sampling + continuous audio chunk DiT denoising."""
        ids_np = np.array(input_ids[0])
        embed_tokens = self.backbone_llm.language_model.embed_tokens
        text_embeds = embed_tokens(input_ids)[0]

        ref_vae_latents_list = []
        # Scatter reference audio VAE latents into <|AUDIO_NO_LATENT|>
        if vae_audios is not None and len(vae_audios) > 0:
            for audio in vae_audios:
                lat = self.red_vae.encode(audio)
                if lat.shape[1] > 0:
                    ref_vae_latents_list.append(lat)
                    patched = self.patch_encoder(lat)[0]
                    indices = np.where(ids_np == self.audio_special_no_latent_id)[0]
                    if len(indices) > 0:
                        start, end = int(indices[0]), int(indices[-1] + 1)
                        text_embeds = mx.concatenate([text_embeds[:start], patched, text_embeds[end:]], axis=0)

        embeds = text_embeds[None, :, :]
        # Prefill
        h, caches = self.backbone_llm.language_model(inputs_embeds=embeds)
        current_h = h[:, -1:]

        # Determine initial mode (text or audio)
        sosp_positions = np.where(ids_np == self.sosp_idx)[0]
        eosp_positions = np.where(ids_np == self.eosp_idx)[0]
        last_sosp_pos = int(sosp_positions[-1]) if len(sosp_positions) > 0 else -1
        last_eosp_pos = int(eosp_positions[-1]) if len(eosp_positions) > 0 else -1
        mode = "audio" if last_sosp_pos > last_eosp_pos else "text"

        if mode == "audio":
            # Keep every prefilled audio-span hidden except the final one.  The
            # hybrid loop appends ``current_h`` below before the first DiT step;
            # including it here duplicates the last conditioning frame and no
            # longer matches the official FireRedAudio generation contract.
            backbone_audio_hiddens = h[:, last_sosp_pos:-1]
        else:
            backbone_audio_hiddens = mx.zeros(
                (1, 0, current_h.shape[-1]), dtype=current_h.dtype
            )

        history_vae_latents = None
        if mode == "audio" and ref_vae_latents_list:
            history_vae_latents = ref_vae_latents_list[-1]

        effective_max_text_tokens = max(max_new_text_tokens, 512) if len(ref_vae_latents_list) == 0 else max(max_new_text_tokens, 256)
        effective_max_audio_steps = max(max_new_audio_steps, 30)

        generated_tokens = []
        generated_latents = []
        n_audio_steps = 0
        n_text_tokens = 0

        while True:
            if mode == "text":
                logits = self.backbone_llm.lm_head(current_h[:, -1, :])
                next_tok = sample_top_k_top_p(logits, 0.7, 20, 0.8)
                generated_tokens.append(next_tok)
                n_text_tokens += 1

                if next_tok == eos_token_id:
                    break

                next_mode = "audio" if next_tok == self.sosp_idx else "text"
                next_embed = embed_tokens(mx.array([[next_tok]]))
                current_h, caches = self.backbone_llm.language_model(
                    inputs_embeds=next_embed,
                    caches=caches,
                )
                mode = next_mode

                if mode == "text" and n_text_tokens >= effective_max_text_tokens:
                    break
            else:
                # Audio mode
                backbone_audio_hiddens = mx.concatenate([backbone_audio_hiddens, current_h[:, -1:]], axis=1)
                one_vae_latents = self.dit.generate(
                    backbone_output=backbone_audio_hiddens,
                    history_vae_latents=history_vae_latents,
                    n_timesteps=n_timesteps,
                    inference_cfg=inference_cfg,
                    solver=solver,
                )
                generated_latents.append(one_vae_latents)
                history_vae_latents = (
                    one_vae_latents
                    if history_vae_latents is None
                    else mx.concatenate([history_vae_latents, one_vae_latents], axis=1)
                )
                n_audio_steps += 1

                next_input_embeds = self.patch_encoder(one_vae_latents)
                current_h, caches = self.backbone_llm.language_model(
                    inputs_embeds=next_input_embeds,
                    caches=caches,
                )

                audio_exit_logits = self.backbone_llm.lm_head(current_h[:, -1, :])
                if n_audio_steps < min_new_audio_steps:
                    audio_exit_logits[:, self.eosp_idx] = -1e9
                next_argmax = int(mx.argmax(audio_exit_logits, axis=-1)[0].item())

                if next_argmax == self.eosp_idx:
                    generated_tokens.append(self.eosp_idx)
                    eosp_embed = embed_tokens(mx.array([[self.eosp_idx]]))
                    current_h, caches = self.backbone_llm.language_model(
                        inputs_embeds=eosp_embed,
                        caches=caches,
                    )
                    mode = "text"
                    backbone_audio_hiddens = mx.zeros(
                        (1, 0, current_h.shape[-1]), dtype=current_h.dtype
                    )
                elif n_audio_steps >= effective_max_audio_steps:
                    break

        if len(generated_latents) > 0:
            all_latents = mx.concatenate(generated_latents, axis=1)
            # Synthesize 24kHz waveform through RedAE Decoder
            audio_waveform = self.vae_decoder(all_latents)[0]
        else:
            all_latents = mx.zeros((1, 0, 64))
            audio_waveform = np.zeros(0, dtype=np.float32)
        return generated_tokens, audio_waveform, all_latents
