#!/usr/bin/env python3
"""Export quantized FireRedAudio model (4-bit or 8-bit) to safetensors directory."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from fireredaudio_mlx import FireRedAudioInference


def quantize_and_export(
    model_dir: str = "models/FireRedAudio",
    output_dir: str = "models/FireRedAudio-8bit",
    bits: int = 8,
    group_size: int = 64,
):
    print("=" * 75)
    print(f"FireRedAudio MLX: Quantizing Model to {bits}-bit (Group Size: {group_size})")
    print("=" * 75)

    src_path = Path(model_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading source model from {src_path}...")
    t0 = time.time()
    engine = FireRedAudioInference(model_path=str(src_path), quantize_bits=bits, quantize_group_size=group_size)
    print(f"Model loaded and quantized in {time.time() - t0:.2f}s")

    print(f"[2/4] Extracting quantized parameters...")
    t0 = time.time()
    weights = dict(tree_flatten(engine.model.parameters()))
    print(f"Extracted {len(weights)} parameter tensors in {time.time() - t0:.2f}s")

    print(f"[3/4] Saving quantized safetensors shards to {out_path}...")
    t0 = time.time()
    # Save weights with safetensors (chunking if needed or direct save)
    shard_name = "model-00001-quant.safetensors"
    mx.save_safetensors(str(out_path / shard_name), weights)
    print(f"Saved weights to {shard_name} ({time.time() - t0:.2f}s)")

    print(f"[4/4] Copying configurations and tokenizer assets...")
    # Copy config and update quantization metadata
    with open(src_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    config["quantization"] = {"bits": bits, "group_size": group_size}
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Copy tokenizer and processor files
    for fname in ["tokenizer.json", "tokenizer_config.json", "processor_config.json"]:
        if (src_path / fname).exists():
            shutil.copyfile(str(src_path / fname), str(out_path / fname))

    # Write index json
    index_data = {
        "metadata": {"total_size": (out_path / shard_name).stat().st_size},
        "weight_map": {k: shard_name for k in weights.keys()}
    }
    with open(out_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    total_mb = (out_path / shard_name).stat().st_size / (1024**2)
    print("=" * 75)
    print(f"🎉 Quantized model exported successfully to: {out_path}")
    print(f"Total weights size on disk: {total_mb:.2f} MB ({total_mb / 1024:.2f} GB)")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/FireRedAudio", help="Source model path")
    parser.add_argument("--output", default="models/FireRedAudio-8bit", help="Output path")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Quantization bits (4 or 8)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    args = parser.parse_args()
    quantize_and_export(args.model, args.output, args.bits, args.group_size)
