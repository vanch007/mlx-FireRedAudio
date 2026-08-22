"""FireRedAudio MLX Native Inference Engine.

Supports all official capabilities:
1. ASR (Speech Recognition)
2. Audio Understanding / QA (with optional CoT thinking)
3. Zero-shot TTS (ICL Voice Cloning)
4. Speech Editing (Semantic text rewrite & Acoustic style/speed modification)
5. Voice Design (Natural language timbre-guided synthesis)
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from .data.processor import FireRedAudioProcessor
from .data.prompt_encoder import (
    AudioPromptEncoder,
    FEAT_TYPE_UNDERSTAND,
    FEAT_TYPE_GENERATION,
    build_understand_prompt,
    build_tts_prompt,
    build_edit_prompt,
    build_voice_design_prompt,
    split_thinking,
    extract_sot_text,
    THINKING_MAX_NEW_TOKENS,
)
from .loader import load_mlx_fireredaudio
from .models.modeling import FireRedAudioModel
from .utils.audio import (
    read_audio,
    pad_to_multiple_of,
    UNDERSTAND_SAMPLE_RATE,
    GENERATION_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

UNDERSTAND_TASKS = ("asr", "understand")
GENERATION_TASKS = ("tts", "edit", "voice_design")
THINKING_TASKS = ("understand",)
DEFAULT_ASR_PROMPT = "Transcribe speech to text."


@dataclass
class UnderstandOutput:
    answer: str
    reasoning: Optional[str] = None


@dataclass
class AudioOutput:
    audio: np.ndarray
    sample_rate: int = GENERATION_SAMPLE_RATE
    text: Optional[str] = None
    vae_latents: Optional[mx.array] = None


class FireRedAudioInference:
    def __init__(
        self,
        model_path: str = "models/FireRedAudio",
        tokenizer_path: Optional[str] = None,
        processor_path: Optional[str] = None,
        quantize_bits: Optional[int] = None,
        quantize_group_size: int = 64,
    ):
        logger.info("Initializing FireRedAudioInference (MLX) from %s...", model_path)
        self.model_path = model_path

        tok_dir = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        self.tokenizer.padding_side = "left"

        proc_dir = processor_path or model_path
        self.processor = FireRedAudioProcessor.from_pretrained(proc_dir)

        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.encoder = AudioPromptEncoder(
            tokenizer=self.tokenizer,
            audio_processor=self.processor,
            audio_special_token=self.config.get("audio_special_token", "<|AUDIO|>"),
            audio_special_token_no_latent=self.config.get("audio_special_token_no_latent", "<|AUDIO_NO_LATENT|>"),
        )

        self._eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self._sot_id = self.tokenizer.convert_tokens_to_ids("<|sot|>")
        self._eot_id = self.tokenizer.convert_tokens_to_ids("<|eot|>")

        t0 = time.time()
        self.weights = load_mlx_fireredaudio(model_path)
        self.load_time = time.time() - t0

        logger.info("Instantiating MLX FireRedAudioModel...")
        self.model = FireRedAudioModel(self.config)

        quant_config = self.config.get("quantization")
        effective_bits = quantize_bits or (quant_config.get("bits") if quant_config else None)
        effective_group_size = quantize_group_size or (quant_config.get("group_size", 64) if quant_config else 64)

        if quant_config:
            self._quantize_backbone(effective_bits, effective_group_size)
            logger.info("Loading %d quantized weights into MLX model...", len(self.weights))
            self.model.load_weights(self.weights)
        else:
            logger.info("Loading %d weights into MLX model...", len(self.weights))
            self.model.load_weights(self.weights)
            if effective_bits in (4, 8):
                self._quantize_backbone(effective_bits, effective_group_size)

        self.quantize_bits = effective_bits
        mx.eval(self.model.parameters())
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        logger.info("FireRedAudio MLX Native Engine ready in %.4fs", self.load_time)

    def _quantize_backbone(self, bits: int, group_size: int = 64):
        def quant_pred(path, m):
            if not isinstance(m, nn.Linear):
                return False
            if "red_vae" in path or "vae_decoder" in path or "audio_encoder" in path:
                return False
            if "backbone_llm" in path:
                return True
            return False

        logger.info("Quantizing 9B Backbone to %d-bit (group_size=%d)...", bits, group_size)
        nn.quantize(self.model, group_size=group_size, bits=bits, class_predicate=quant_pred)

    def understand(
        self,
        audio_paths: Union[str, List[str]],
        prompt: str = DEFAULT_ASR_PROMPT,
        task: str = "understand",
        enable_thinking: bool = False,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> UnderstandOutput:
        """Run speech recognition or audio understanding with MLX neural forward passes."""
        if enable_thinking and task not in THINKING_TASKS:
            raise ValueError(f"task={task!r} does not support enable_thinking.")
        if max_new_tokens is None:
            max_new_tokens = THINKING_MAX_NEW_TOKENS if enable_thinking else 300
        if num_beams is None:
            num_beams = 4 if task == "asr" else 1
        if temperature is None:
            temperature = 0.0 if task == "asr" else 0.7
        if top_k is None:
            top_k = 20
        if top_p is None:
            top_p = 0.8

        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        input_chatml = build_understand_prompt(
            prompt,
            num_audios=len(audio_paths),
            audio_special_token=self.config.get("audio_special_token", "<|AUDIO|>"),
            enable_thinking=enable_thinking,
        )

        input_audios = [{
            "feat_type": FEAT_TYPE_UNDERSTAND,
            "audio_understand": read_audio(p, UNDERSTAND_SAMPLE_RATE),
            "audio_generation": None,
            "role": "user",
        } for p in audio_paths]

        batch = self.encoder.encode(input_chatml, input_audios)

        input_ids = mx.array(batch["input_ids"])
        if batch.get("audio_features") is not None and len(batch["audio_features"]) > 0:
            audio_features = []
            for index in range(len(batch["audio_features"])):
                mel_len = int(batch["audio_feature_attention_mask"][index].sum())
                audio_features.append(mx.array(batch["audio_features"][index, :, :mel_len]))
        else:
            audio_features = None

        gen_tokens = self.model.generate(
            input_ids=input_ids,
            audio_features=audio_features,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            eos_token_id=self._eos_id,
        )

        decoded_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
        for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            decoded_text = decoded_text.replace(token, "")
        decoded_text = decoded_text.strip()

        if enable_thinking:
            reasoning, answer = split_thinking(decoded_text)
            return UnderstandOutput(answer=answer or decoded_text, reasoning=reasoning)
        else:
            return UnderstandOutput(answer=decoded_text, reasoning=None)

    def tts(
        self,
        prompt_text: str,
        prompt_audio: str,
        target_text: str,
        language: str = "zh",
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 6,
        max_new_text_tokens: int = 512,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ) -> AudioOutput:
        """Zero-shot TTS voice cloning via MLX hybrid AR generation."""
        input_chatml = build_tts_prompt(
            prompt_text,
            target_text,
            language,
            self.config.get("audio_special_token_no_latent", "<|AUDIO_NO_LATENT|>"),
        )
        ref_audio = pad_to_multiple_of(read_audio(prompt_audio, GENERATION_SAMPLE_RATE))
        input_audios = [{
            "feat_type": FEAT_TYPE_GENERATION,
            "audio_understand": None,
            "audio_generation": ref_audio,
            "role": "assistant",
        }]

        batch = self.encoder.encode(input_chatml, input_audios)
        input_ids = mx.array(batch["input_ids"])

        gen_tokens, audio_waveform, vae_latents = self.model.generate_tts(
            input_ids=input_ids,
            vae_audios=[ref_audio],
            max_new_audio_steps=max_new_audio_steps,
            min_new_audio_steps=min_new_audio_steps,
            max_new_text_tokens=max_new_text_tokens,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            eos_token_id=self._eos_id,
        )

        if len(audio_waveform) == 0:
            raise RuntimeError("task='tts' produced no audio latents")
        return AudioOutput(
            audio=audio_waveform.astype(np.float32),
            sample_rate=GENERATION_SAMPLE_RATE,
            text=target_text,
            vae_latents=vae_latents,
        )

    def edit(
        self,
        audio_path: str,
        instruction: str,
        edit_type: str = "acoustic",
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 6,
        max_new_text_tokens: int = 512,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ) -> AudioOutput:
        """Speech editing in MLX."""
        input_chatml = build_edit_prompt(
            instruction,
            edit_type=edit_type,
            audio_no_latent_token=self.config.get("audio_special_token_no_latent", "<|AUDIO_NO_LATENT|>"),
        )
        src_audio = pad_to_multiple_of(read_audio(audio_path, GENERATION_SAMPLE_RATE))
        input_audios = [{
            "feat_type": FEAT_TYPE_GENERATION,
            "audio_understand": None,
            "audio_generation": src_audio,
            "role": "user",
        }]

        batch = self.encoder.encode(input_chatml, input_audios)
        input_ids = mx.array(batch["input_ids"])

        gen_tokens, audio_waveform, vae_latents = self.model.generate_tts(
            input_ids=input_ids,
            vae_audios=[src_audio],
            max_new_audio_steps=max_new_audio_steps,
            min_new_audio_steps=min_new_audio_steps,
            max_new_text_tokens=max_new_text_tokens,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            eos_token_id=self._eos_id,
        )

        if len(audio_waveform) == 0:
            raise RuntimeError(f"task='edit' ({edit_type}) produced no audio latents")
        sot_text = extract_sot_text(gen_tokens, self.tokenizer, self._sot_id, self._eot_id)
        rewritten_text = sot_text if edit_type == "semantic" else None
        return AudioOutput(
            audio=audio_waveform.astype(np.float32),
            sample_rate=GENERATION_SAMPLE_RATE,
            text=rewritten_text,
            vae_latents=vae_latents,
        )

    def voice_design(
        self,
        instruction: str,
        text: str,
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 6,
        max_new_text_tokens: int = 512,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ) -> AudioOutput:
        """Speech synthesis guided by natural language timbre description in MLX."""
        input_chatml = build_voice_design_prompt(
            instruction,
            text,
            audio_no_latent_token=self.config.get("audio_special_token_no_latent", "<|AUDIO_NO_LATENT|>"),
        )

        batch = self.encoder.encode(input_chatml, audios=[])
        input_ids = mx.array(batch["input_ids"])

        gen_tokens, audio_waveform, vae_latents = self.model.generate_tts(
            input_ids=input_ids,
            vae_audios=[],
            max_new_audio_steps=max_new_audio_steps,
            min_new_audio_steps=min_new_audio_steps,
            max_new_text_tokens=max_new_text_tokens,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            eos_token_id=self._eos_id,
        )

        if len(audio_waveform) == 0:
            raise RuntimeError("task='voice_design' produced no audio latents")
        sot_text = extract_sot_text(gen_tokens, self.tokenizer, self._sot_id, self._eot_id)
        return AudioOutput(
            audio=audio_waveform.astype(np.float32),
            sample_rate=GENERATION_SAMPLE_RATE,
            text=sot_text or f"<|timbre|>{instruction}<|/timbre|>",
            vae_latents=vae_latents,
        )
