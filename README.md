# mlx-FireRedAudio 🎙️⚡

Native Apple Silicon MLX port of **FireRedAudio**: A General-Purpose Audio Language Model with Decoupled Continuous Representations for Understanding and Generation.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![MLX](https://img.shields.io/badge/MLX-0.32.1-red.svg)](https://github.com/ml-explore/mlx)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/vanch007)

---

## Highlights 🚀

- ⚡ **Apple Silicon Metal Accelerated**: MLX unified-memory weights, fused Metal SDPA attention, and dedicated Gated Delta Metal kernels.
- 🚀 **Real-Time Factor (RTF) Breakthrough**:
  - **ASR (Speech Recognition)**: **RTF `0.10 ~ 0.14`** (7.2x ~ 9.6x real-time speedup)
  - **TTS (Voice Cloning)**: **RTF `0.68 ~ 0.75`** (Sub-second latency, true real-time speech generation)
- 🎙️ **Full Audio Stack (5 Core Capabilities)**:
  1. **Speech Recognition (ASR)**: High-accuracy multilingual speech transcription.
  2. **Audio Understanding / QA**: Audio reasoning with optional Chain-of-Thought (CoT) thinking.
  3. **Zero-shot Voice Cloning (TTS)**: Timbre continuation from prompt audio and transcript.
  4. **Speech Editing**: Semantic content rewrites and acoustic rate/pitch/volume editing.
  5. **Voice Design**: Natural language timbre-guided audio generation.
- 🌐 **FireRedAudio Studio WebUI**: Modern FastAPI + React 18 / Vite SPA with real-time SSE streaming, single-instance job queue, project management, and Metal memory monitoring.
- 🧩 **100% Native Inference Stack**: Zero PyTorch, CUDA, MPS, TensorFlow, or JAX runtime dependencies. Neural forward passes and ISTFT run natively in MLX.

---

## 📊 Benchmark & Quantization Matrix

Tested on Apple M3 Max (Metal Unified Memory):

| Checkpoint / Precision | Memory Footprint | ASR (9.6s Audio) | ASR RTF | TTS (3.5s Speech) | TTS RTF | Hugging Face Model |
|:---|:---|:---|:---|:---|:---|:---|
| **8-bit (Recommended)** | **13.54 GB** | **1.34s** | **`0.1396`** | **2.62s** | **`0.7529`** | [🤗 vanch007/FireRedAudio-MLX-8bit](https://huggingface.co/vanch007/FireRedAudio-MLX-8bit) |
| **4-bit (Fastest)** | **9.85 GB** | **1.00s** | **`0.1037`** | **2.28s** | **`0.6794`** | [🤗 vanch007/FireRedAudio-MLX-4bit](https://huggingface.co/vanch007/FireRedAudio-MLX-4bit) |
| **BF16 (Original)** | 20.47 GB | 6.28s | `0.6546` | 13.91s | `3.2209` | [🤗 FireRedTeam/FireRedAudio](https://huggingface.co/FireRedTeam/FireRedAudio) |

---

## 📦 Installation

```bash
git clone https://github.com/vanch007/mlx-FireRedAudio.git
cd mlx-FireRedAudio
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### Download Pre-Quantized Models

```bash
# 8-bit Real-time Model (Recommended)
hf download vanch007/FireRedAudio-MLX-8bit --local-dir models/FireRedAudio-8bit

# 4-bit Ultra-fast Model (Lightweight 16GB Macs)
hf download vanch007/FireRedAudio-MLX-4bit --local-dir models/FireRedAudio-4bit
```

---

## 🌐 WebUI Studio (本地工作流中心)

FireRedAudio Studio 提供开箱即用的现代化 WebUI（FastAPI + React/Vite SPA），集成了四大核心能力工作台、项目管理、声音模板库、排队调度与 Apple Silicon Metal 显存实时监控：

```bash
# 启动 8-bit 加速 WebUI 本地服务 (默认端口 7860)
python run_webui.py --model models/FireRedAudio-8bit

# 或使用原始 BF16 模型
python run_webui.py --model models/FireRedAudio
```

打开浏览器访问：**http://127.0.0.1:7860**
- 交互式 OpenAPI 文档：**http://127.0.0.1:7860/docs**
- 实时事件流：`GET /api/v1/events` (Server-Sent Events)

---

## 💻 Python API

```python
from fireredaudio_mlx import FireRedAudioInference

# Load 8-bit quantized model
engine = FireRedAudioInference(model_path="models/FireRedAudio-8bit")

# 1. ASR Speech Transcription (RTF ~ 0.14)
res = engine.understand("assets/examples/asr_zh_fleurs.wav", task="asr")
print("ASR Transcript:", res.answer)

# 2. Audio Understanding with CoT Thinking
res_qa = engine.understand(
    "assets/examples/two_speakers.wav",
    prompt="这个音频中有几个说话人？",
    task="understand",
    enable_thinking=True,
)
print("Thinking Reasoning:", res_qa.reasoning)
print("Answer:", res_qa.answer)

# 3. Zero-shot Voice Cloning TTS (RTF ~ 0.75)
audio_res = engine.tts(
    prompt_text="同时，他强调微调要科学有序。",
    prompt_audio="assets/examples/tts_zh_prompt.wav",
    target_text="你好，欢迎体验 FireRedAudio MLX 实时语音大模型！",
)

# 4. Speech Editing (Acoustic / Semantic)
edit_res = engine.edit(
    audio_path="assets/examples/edit_semantic_zh_ref.wav",
    instruction="delete '比普通的茶叶要'",
    edit_type="semantic",
)

# 5. Voice Design
vd_res = engine.voice_design(
    instruction="温柔知性的广播女主播声音",
    text="欢迎收听今日新闻，我们将为您带来最新的科技前沿报道。",
)
```

---

## 🛠️ CLI 命令行推理

```bash
# 1. ASR 转写
python inference.py --task asr --audio assets/examples/asr_zh_fleurs.wav

# 2. 音频问答 (Thinking)
python inference.py --task understand --audio assets/examples/two_speakers.wav --prompt "这个音频中有几个说话人？" --enable-thinking

# 3. 声音克隆 (TTS)
python inference.py --task tts --prompt-audio assets/examples/tts_zh_prompt.wav --prompt-text "同时，他强调微调要科学有序。" --target-text "你好，欢迎使用 MLX 原生版本！" --output outputs/tts_output.wav

# 4. 语音编辑 (Acoustic / Semantic)
python inference.py --task edit --audio assets/examples/edit_acoustic_zh_ref.wav --instruction "adjust the speed to 0.5" --edit-type acoustic --output outputs/edit_slow.wav

# 5. 音色设计 (Voice Design)
python inference.py --task voice_design --instruction "温柔清晰的播音女声" --text "这是通过音色描述直接生成的语音。" --output outputs/voice_design.wav
```

---

## 🧪 Testing & Verification

```bash
# 运行单元与 API 测试套件（当前 24 项）
.venv/bin/python -m unittest discover -s tests -v

# 内容级生成回归：TTS + Voice Design，并用本项目 ASR 校验 CER
.venv/bin/python scripts/test_generation_content_regression.py

# 运行 10 连续任务队列与单实例内存压力测试
.venv/bin/python scripts/test_webui_consecutive_stress.py
```

---

## 📄 License & Acknowledgments

- This project is released under the [Apache 2.0 License](LICENSE).
- Based on [FireRedAudio](https://github.com/FireRedTeam/FireRedAudio) by Xiaohongshu FireRedTeam.
- Powered by Apple [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm).
