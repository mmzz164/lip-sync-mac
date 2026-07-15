#!/bin/bash
# Start ComfyUI on Apple Silicon for LTX-2.3 GGUF lip-sync generation.
# Usage: bash run_comfyui_mac.sh
# Stop : Ctrl-C, or from another terminal: pkill -f "main.py --listen 127.0.0.1 --port 8188"
ROOT="${LTX_MAC_ROOT:-$HOME/work/ltx_mac}"
export PATH=/opt/homebrew/bin:$PATH
# Raise the MPS memory watermarks so PyTorch can use more of unified memory
# without leaving the whole OS to fight over the rest (0.0 = unlimited is
# unsafe here - it can pressure the OS into a hard freeze on some Macs).
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.85}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.75}
cd "$ROOT/ComfyUI"
exec "$ROOT/venv/bin/python" main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --base-directory "$ROOT" \
    --output-directory "$ROOT/output" \
    --input-directory "$ROOT/input" \
    --preview-method none "$@"
