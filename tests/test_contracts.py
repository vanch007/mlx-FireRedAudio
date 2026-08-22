import unittest

import mlx.core as mx
import numpy as np

from fireredaudio_mlx.data.prompt_encoder import (
    build_edit_prompt,
    build_tts_prompt,
    split_thinking,
)
from fireredaudio_mlx.models.modeling import sanitize_weights
from fireredaudio_mlx.models.audio_encoder import FireRedAudioEncoderMLX
from fireredaudio_mlx.models.redae import ISTFTHeadMLX


class PromptContractTests(unittest.TestCase):
    def test_thinking_output_starts_inside_open_prompt_block(self):
        reasoning, answer = split_thinking("先分析问题\n</think>\n\n最终答案")
        self.assertEqual(reasoning, "先分析问题")
        self.assertEqual(answer, "最终答案")

    def test_tts_prompt_ends_in_audio_mode(self):
        prompt = build_tts_prompt("参考", "目标")
        self.assertTrue(prompt.endswith("<|sosp|><|AUDIO_NO_LATENT|>"))

    def test_edit_prompt_starts_generation_in_text_mode(self):
        prompt = build_edit_prompt("adjust the speed to 0.5", "acoustic")
        self.assertIn("<|sosp|><|AUDIO_NO_LATENT|><|eosp|>", prompt)
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))


class WeightSanitizationTests(unittest.TestCase):
    def test_qwen35_zero_centered_norm_is_shifted(self):
        key = "backbone_llm.model.language_model.layers.0.input_layernorm.weight"
        result = sanitize_weights({key: mx.array([0.0, 0.25])})
        mapped = "backbone_llm.language_model.layers.0.input_layernorm.weight"
        self.assertEqual(result[mapped].tolist(), [1.0, 1.25])

    def test_redae_norm_is_not_shifted(self):
        key = "red_vae.qwen3.layers.0.input_layernorm.weight"
        result = sanitize_weights({key: mx.array([0.75, 1.0])})
        self.assertEqual(result[key].tolist(), [0.75, 1.0])


class NativeISTFTTests(unittest.TestCase):
    def test_mlx_overlap_add_matches_numpy_reference(self):
        rng = np.random.default_rng(0)
        real = rng.normal(size=(1, 5, 961)).astype(np.float32)
        imag = rng.normal(size=(1, 5, 961)).astype(np.float32)
        head = ISTFTHeadMLX()
        actual = np.array(head.decode_waveform(mx.array(real), mx.array(imag)))

        window = np.hanning(1921)[:-1].astype(np.float32)
        frames = np.fft.irfft(real + 1j * imag, n=1920, axis=-1) * window
        expected = np.zeros((1, 3840), dtype=np.float32)
        envelope = np.zeros(3840, dtype=np.float32)
        for frame_index in range(5):
            start = frame_index * 480
            expected[:, start : start + 1920] += frames[:, frame_index]
            envelope[start : start + 1920] += window**2
        expected = np.divide(
            expected,
            envelope[None, :],
            out=np.zeros_like(expected),
            where=envelope[None, :] > 1e-11,
        )[:, 720:-720]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class LongAudioChunkingTests(unittest.TestCase):
    def test_audio_encoder_chunks_beyond_window(self):
        encoder = FireRedAudioEncoderMLX({
            "d_model": 8,
            "output_dim": 8,
            "encoder_layers": 0,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "max_source_positions": 4,
            "n_window": 4,
        })
        # 17 mel frames cross two full 8-frame windows plus a tail.
        output = encoder(mx.zeros((128, 17)))
        self.assertEqual(output.shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
