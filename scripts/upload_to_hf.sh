#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "Uploading FireRedAudio Quantized Models to Hugging Face (vanch007)"
echo "=========================================================================="

echo ""
echo "[1/2] Uploading 8-bit model to vanch007/FireRedAudio-MLX-8bit..."
hf upload-large-folder vanch007/FireRedAudio-MLX-8bit models/FireRedAudio-8bit --repo-type model

echo ""
echo "[2/2] Uploading 4-bit model to vanch007/FireRedAudio-MLX-4bit..."
hf upload-large-folder vanch007/FireRedAudio-MLX-4bit models/FireRedAudio-4bit --repo-type model

echo ""
echo "=========================================================================="
echo "🎉 All models uploaded successfully to Hugging Face!"
echo "  - https://huggingface.co/vanch007/FireRedAudio-MLX-8bit"
echo "  - https://huggingface.co/vanch007/FireRedAudio-MLX-4bit"
echo "=========================================================================="
