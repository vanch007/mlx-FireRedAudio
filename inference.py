#!/usr/bin/env python3
"""FireRedAudio MLX Inference CLI.

Commands:
    # 1. Speech recognition (ASR)
    python inference.py --task asr --model models/FireRedAudio --audio assets/examples/asr_zh_fleurs.wav

    # 2. Audio understanding / QA (with optional thinking reasoning)
    python inference.py --task understand --model models/FireRedAudio --audio assets/examples/two_speakers.wav \
        --prompt "这个音频中有几个说话人？" --enable-thinking

    # 3. Zero-shot Voice Cloning (TTS)
    python inference.py --task tts --model models/FireRedAudio \
        --prompt-audio assets/examples/tts_zh_prompt.wav --prompt-text "收到你的来信，我很高兴。" \
        --target-text "你好，欢迎使用 FireRedAudio MLX 版本！" --output outputs/tts.wav

    # 4. Speech Editing
    python inference.py --task edit --model models/FireRedAudio \
        --audio assets/examples/edit_acoustic_zh_ref.wav --instruction "adjust the speed to 0.5" --edit-type acoustic

    # 5. Voice Design
    python inference.py --task voice_design --model models/FireRedAudio \
        --instruction "温柔清晰的播音女声" --text "这是通过音色描述直接生成的语音。"
"""

import argparse
import logging
import os
import sys
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fireredaudio_mlx.inference import FireRedAudioInference, UNDERSTAND_TASKS, GENERATION_TASKS, THINKING_TASKS

logger = logging.getLogger("fireredaudio_mlx")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(UNDERSTAND_TASKS) + list(GENERATION_TASKS))
    p.add_argument("--model", default="models/FireRedAudio", help="Directory holding MLX model files")
    p.add_argument("--tokenizer", default=None, help="Defaults to --model")
    p.add_argument("--processor", default=None, help="Defaults to --model")
    p.add_argument("--vae-decoder", default=None, help="Compatibility flag (MLX has integrated decoder)")
    p.add_argument("--device", default="mlx", help="Device target (MLX Metal default)")
    p.add_argument("--output", default=None, help="Output file path (txt for understanding, wav for generation)")

    # Understanding options
    p.add_argument("--audio", nargs="+", default=None, help="Input audio file(s) for understanding or editing")
    p.add_argument("--prompt", default=None, help="Prompt text for audio understanding")
    p.add_argument("--enable-thinking", action="store_true", help="Enable Chain-of-Thought thinking for understand task")

    # TTS options
    p.add_argument("--prompt-audio", default=None, help="Reference audio for voice cloning")
    p.add_argument("--prompt-text", default=None, help="Transcript of reference audio")
    p.add_argument("--target-text", default=None, help="Target text to synthesize")
    p.add_argument("--language", default="zh", choices=["zh", "en"])

    # Editing / Voice design options
    p.add_argument("--instruction", default=None, help="Editing instruction or voice design description")
    p.add_argument("--edit-type", default="acoustic", choices=["semantic", "acoustic"])
    p.add_argument("--text", default=None, help="Text for voice design to synthesize")

    # Common generation hyperparams
    p.add_argument("--max-new-audio-steps", type=int, default=750)
    p.add_argument("--min-new-audio-steps", type=int, default=6)
    p.add_argument("--max-new-text-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--inference-cfg", type=float, default=2.0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    engine = FireRedAudioInference(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        processor_path=args.processor,
    )

    if args.task in UNDERSTAND_TASKS:
        if not args.audio:
            raise SystemExit(f"--task {args.task} requires --audio <path>")
        prompt = args.prompt or ("Transcribe speech to text." if args.task == "asr" else "What is in this audio?")
        res = engine.understand(
            audio_paths=args.audio,
            prompt=prompt,
            task=args.task,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )
        if res.reasoning:
            print("===== Thinking / Reasoning =====")
            print(res.reasoning)
        print("===== Output =====")
        print(res.answer)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                if res.reasoning:
                    f.write("<think>\n" + res.reasoning + "\n</think>\n")
                f.write(res.answer + "\n")
            logger.info("Saved text output to %s", args.output)

    elif args.task == "tts":
        if not args.prompt_audio or not args.prompt_text or not args.target_text:
            raise SystemExit("--task tts requires --prompt-audio, --prompt-text, and --target-text")
        res = engine.tts(
            prompt_text=args.prompt_text,
            prompt_audio=args.prompt_audio,
            target_text=args.target_text,
            language=args.language,
            max_new_audio_steps=args.max_new_audio_steps,
            min_new_audio_steps=args.min_new_audio_steps,
            max_new_text_tokens=args.max_new_text_tokens,
            n_timesteps=args.n_timesteps,
            inference_cfg=args.inference_cfg,
        )
        out_wav = args.output or "outputs/tts_output.wav"
        os.makedirs(os.path.dirname(out_wav) if os.path.dirname(out_wav) else ".", exist_ok=True)
        sf.write(out_wav, res.audio, res.sample_rate, subtype="PCM_16")
        dur = len(res.audio) / res.sample_rate
        logger.info("Saved synthesized audio to %s (duration: %.2fs)", out_wav, dur)

    elif args.task == "edit":
        if not args.audio or not args.instruction:
            raise SystemExit("--task edit requires --audio and --instruction")
        res = engine.edit(
            audio_path=args.audio[0],
            instruction=args.instruction,
            edit_type=args.edit_type,
            max_new_audio_steps=args.max_new_audio_steps,
            min_new_audio_steps=args.min_new_audio_steps,
            max_new_text_tokens=args.max_new_text_tokens,
            n_timesteps=args.n_timesteps,
            inference_cfg=args.inference_cfg,
        )
        out_wav = args.output or f"outputs/edit_{args.edit_type}.wav"
        os.makedirs(os.path.dirname(out_wav) if os.path.dirname(out_wav) else ".", exist_ok=True)
        sf.write(out_wav, res.audio, res.sample_rate, subtype="PCM_16")
        if res.text:
            print("===== Edited Text =====")
            print(res.text)
        logger.info("Saved edited audio to %s (duration: %.2fs)", out_wav, len(res.audio) / res.sample_rate)

    elif args.task == "voice_design":
        if not args.instruction or not args.text:
            raise SystemExit("--task voice_design requires --instruction and --text")
        res = engine.voice_design(
            instruction=args.instruction,
            text=args.text,
            max_new_audio_steps=args.max_new_audio_steps,
            min_new_audio_steps=args.min_new_audio_steps,
            max_new_text_tokens=args.max_new_text_tokens,
            n_timesteps=args.n_timesteps,
            inference_cfg=args.inference_cfg,
        )
        out_wav = args.output or "outputs/voice_design.wav"
        os.makedirs(os.path.dirname(out_wav) if os.path.dirname(out_wav) else ".", exist_ok=True)
        sf.write(out_wav, res.audio, res.sample_rate, subtype="PCM_16")
        if res.text:
            print("===== Timbre Description =====")
            print(res.text)
        logger.info("Saved voice design audio to %s (duration: %.2fs)", out_wav, len(res.audio) / res.sample_rate)


if __name__ == "__main__":
    main()
