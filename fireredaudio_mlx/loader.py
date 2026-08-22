"""Weights loader for MLX sharded safetensors."""

import json
import os
import time
import logging
from typing import Dict
import mlx.core as mx

logger = logging.getLogger(__name__)


def load_mlx_fireredaudio(model_dir: str = "models/FireRedAudio") -> Dict[str, mx.array]:
    """Load all MLX safetensors shards using zero-copy memory mapping."""
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Model index file not found: {index_file}")

    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    config_file = os.path.join(model_dir, "config.json")
    is_mlx_native = False
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "quantization" in cfg or cfg.get("framework") == "mlx":
                    is_mlx_native = True
        except Exception:
            pass

    weight_map = data.get("weight_map", {})
    shards = sorted(list(set(weight_map.values())))

    t0 = time.time()
    weights: Dict[str, mx.array] = {}
    for shard in shards:
        shard_path = os.path.join(model_dir, shard)
        if os.path.exists(shard_path):
            w = mx.load(shard_path)
            weights.update(w)
        else:
            logger.warning("Shard not found: %s", shard_path)

    dur = time.time() - t0
    logger.info("Loaded %d tensors across %d shards in %.4fs (is_mlx_native=%s)", len(weights), len(shards), dur, is_mlx_native)
    return weights
