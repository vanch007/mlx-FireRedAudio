"""Prompt and ChatML encoder for FireRedAudio."""

import math
import re
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from ..utils.audio import UNDERSTAND_SAMPLE_RATE, VAE_DOWNSAMPLE_RATE, PATCH_ENCODER_DOWNSAMPLE_RATE
from .processor import FireRedAudioProcessor

FEAT_TYPE_UNDERSTAND = "feat_understand"
FEAT_TYPE_GENERATION = "feat_generation"

GENERIC_SYSTEM_PROMPT = "You are a helpful assistant."
UNDERSTAND_SYSTEM_PROMPT = (
    "You are an audio understanding expert. Please answer user questions based on the audio."
)
THINKING_MAX_NEW_TOKENS = 2048


def split_thinking(text: str) -> Tuple[Optional[str], str]:
    # The opening <think> token is part of the prompt, so generation starts
    # inside the block and only emits the closing marker.
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.strip(), answer.strip()
    return None, text.strip()


def extract_sot_text(token_ids: List[int], tokenizer, sot_id: int, eot_id: int) -> Optional[str]:
    if sot_id in token_ids and eot_id in token_ids:
        s_idx = token_ids.index(sot_id)
        e_idx = token_ids.index(eot_id)
        if s_idx < e_idx:
            sub = token_ids[s_idx + 1 : e_idx]
            return tokenizer.decode(sub, skip_special_tokens=True).strip()
    return None


def _chatml(
    system: str, user: str, assistant_prefix: str = "", enable_thinking: bool = False
) -> str:
    if enable_thinking and assistant_prefix:
        raise ValueError("enable_thinking cannot be combined with assistant_prefix")
    think = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{think}{assistant_prefix}"
    )


def build_understand_prompt(
    prompt: str,
    num_audios: int,
    audio_special_token: str = "<|AUDIO|>",
    enable_thinking: bool = False,
) -> str:
    sys_msg = UNDERSTAND_SYSTEM_PROMPT
    audio_segs = "".join(
        f"Audio {i + 1}: <|sosp|>{audio_special_token}<|eosp|>\n"
        for i in range(num_audios)
    )
    return _chatml(sys_msg, f"{audio_segs}{prompt}", enable_thinking=enable_thinking)


def build_tts_prompt(
    prompt_text: str,
    target_text: str,
    language: str = "zh",
    audio_no_latent_token: str = "<|AUDIO_NO_LATENT|>",
) -> str:
    sep = " " if language == "en" else ""
    return _chatml(
        GENERIC_SYSTEM_PROMPT,
        f"Convert text to speech.\n{prompt_text}{sep}{target_text}",
        assistant_prefix=f"<|sosp|>{audio_no_latent_token}",
    )


def build_edit_prompt(
    instruction: str,
    edit_type: str = "acoustic",
    audio_no_latent_token: str = "<|AUDIO_NO_LATENT|>",
) -> str:
    if edit_type == "semantic":
        user_text = f"Identify the content of the audio. {instruction}"
    elif edit_type == "acoustic":
        user_text = instruction
    else:
        raise ValueError(f"unknown edit_type {edit_type!r}, expected semantic or acoustic")
    return _chatml(
        GENERIC_SYSTEM_PROMPT,
        f"Audio 1: <|sosp|>{audio_no_latent_token}<|eosp|>\n{user_text}",
    )


def build_voice_design_prompt(
    instruction: str,
    text: str,
    audio_no_latent_token: str = "<|AUDIO_NO_LATENT|>",
) -> str:
    prompt_body = f"{instruction}\n\n根据上述音色描述，合成以下文本对应的音频：\n{text}"
    return _chatml(
        GENERIC_SYSTEM_PROMPT,
        prompt_body,
        assistant_prefix="<|sosp|>",
    )


class AudioPromptEncoder:
    def __init__(
        self,
        tokenizer,
        audio_processor: FireRedAudioProcessor,
        audio_special_token: str = "<|AUDIO|>",
        audio_special_token_no_latent: str = "<|AUDIO_NO_LATENT|>",
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.audio_special_token = audio_special_token
        self.audio_special_token_no_latent = audio_special_token_no_latent

    @staticmethod
    def get_audio_output_lengths(feat_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        conv2_len = (feat_lens - 1) // 2 + 1
        conv3_len = (conv2_len - 1) // 2 + 1
        conv4_len = (conv3_len - 1) // 2 + 1
        return conv2_len, conv4_len

    def pad_and_mask(self, tensor_list, padding_value=0.0):
        if not tensor_list:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=bool)
        lengths = [len(t) // PATCH_ENCODER_DOWNSAMPLE_RATE for t in tensor_list]
        max_samples = max(len(t) for t in tensor_list)
        max_len = max(lengths) if lengths else 0
        padded = np.zeros((len(tensor_list), max_samples), dtype=np.float32)
        mask = np.zeros((len(tensor_list), max_len), dtype=bool)
        for i, (t, l) in enumerate(zip(tensor_list, lengths)):
            padded[i, : len(t)] = t
            mask[i, :l] = True
        return padded, mask

    def encode(self, chatml: str, audios: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_arrays = []
        vae_audio_arrays = []
        vae_is_assistant = []

        for audio in audios:
            if audio["feat_type"] == FEAT_TYPE_UNDERSTAND:
                audio_arrays.append(audio["audio_understand"])
            elif audio["feat_type"] == FEAT_TYPE_GENERATION:
                vae_audio_arrays.append(audio["audio_generation"])
                vae_is_assistant.append(audio.get("role") == "assistant")

        # Understanding feature extraction
        if audio_arrays:
            max_len_samples = max(len(a) for a in audio_arrays) + UNDERSTAND_SAMPLE_RATE
            res = self.audio_processor(audios=audio_arrays, max_length=max_len_samples)
            audio_features = res["input_features"]
            audio_feature_mask = res["feature_attention_mask"]
            _, out_lens = self.get_audio_output_lengths(audio_feature_mask.sum(axis=-1))
            replace_lens = out_lens.tolist()
        else:
            audio_features = np.empty((0, 128, 1), dtype=np.float32)
            audio_feature_mask = np.empty((0, 1), dtype=np.int32)
            replace_lens = []

        # Generation feature lengths (25Hz / 4 = 6.25Hz)
        vae_replace_lens = []
        for vae_a in vae_audio_arrays:
            patch_len = len(vae_a) // PATCH_ENCODER_DOWNSAMPLE_RATE
            vae_replace_lens.append(patch_len)

        # Placeholders expansion with sentinel tokens to avoid infinite replacement loop
        sentinel = "__FIREREDAUDIO_PLACEHOLDER__"
        expanded_chatml = chatml

        # Replace understanding placeholders
        for rep_len in replace_lens:
            expanded_chatml = expanded_chatml.replace(self.audio_special_token, sentinel, 1)
            expanded_chatml = expanded_chatml.replace(sentinel, self.audio_special_token * rep_len, 1)

        # Replace generation placeholders
        for rep_len in vae_replace_lens:
            expanded_chatml = expanded_chatml.replace(self.audio_special_token_no_latent, sentinel, 1)
            expanded_chatml = expanded_chatml.replace(sentinel, self.audio_special_token_no_latent * rep_len, 1)

        input_ids = self.tokenizer.encode(expanded_chatml, add_special_tokens=False)
        input_ids = np.array([input_ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int32)

        padded_vae_audios, patch_mask = self.pad_and_mask(vae_audio_arrays)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_features": audio_features,
            "audio_feature_attention_mask": audio_feature_mask,
            "vae_audios": padded_vae_audios,
            "patch_encoder_output_attention_mask": patch_mask,
            "vae_is_assistant": np.array(vae_is_assistant, dtype=bool),
            "expanded_chatml": expanded_chatml,
        }
