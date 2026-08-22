"""Configuration and presets for FireRedAudio WebUI."""

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_WORKSPACE_DIR = REPO_ROOT / "workspace"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "FireRedAudio-8bit"
FRONTEND_DIST_DIR = PACKAGE_DIR / "frontend" / "dist"

QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
   "fast": {
       "id": "fast",
       "name": "快速 (Fast)",
       "description": "极速生成与实时响应，Flow 4 步，ASR 快速模式",
       "n_timesteps": 4,
       "asr_beam_size": 1,
       "temperature": 0.7,
       "top_k": 20,
       "top_p": 0.8,
        "inference_cfg": 1.5,
   },
   "balanced": {
       "id": "balanced",
       "name": "平衡 (Balanced - M3 Max 推荐)",
       "description": "高品质且响应迅速，Flow 4 步，ASR 官方 4 束精准模式",
       "n_timesteps": 4,
       "asr_beam_size": 4,
       "temperature": 0.7,
       "top_k": 20,
       "top_p": 0.8,
        "inference_cfg": 1.5,
   },
   "high_quality": {
       "id": "high_quality",
       "name": "高质量 (High Quality)",
       "description": "官方最高品质，Flow 10 步细致去噪积分，ASR 4 束搜索",
       "n_timesteps": 10,
       "asr_beam_size": 4,
       "temperature": 0.7,
       "top_k": 20,
       "top_p": 0.8,
        "inference_cfg": 1.5,
   },
}
