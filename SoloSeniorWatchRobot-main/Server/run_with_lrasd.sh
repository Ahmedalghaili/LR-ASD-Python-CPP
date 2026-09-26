#!/usr/bin/env bash
set -euo pipefail

export LR_ASD_MODEL_DIR="${LR_ASD_MODEL_DIR:-$HOME/SoloSeniorWatchRobot_build/lr-asd}"
project_root="$(cd "$(dirname "$0")/../.." && pwd)"
scrfd_model_default="$LR_ASD_MODEL_DIR/scrfd_2.5g_bnkps.dynamic.onnx"
if [[ ! -f "$scrfd_model_default" && -f "$project_root/experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx" ]]; then
    scrfd_model_default="$project_root/experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx"
fi
# Select the face detector without rebuilding. Use s3fd for the original
# backend or scrfd for SCRFD-2.5G. Override the model path when the model is
# stored outside LR_ASD_MODEL_DIR.
export LR_ASD_DETECTOR="${LR_ASD_DETECTOR:-s3fd}"
export LR_ASD_SCRFD_MODEL="${LR_ASD_SCRFD_MODEL:-$scrfd_model_default}"
export LR_ASD_S3FD_MODEL="${LR_ASD_S3FD_MODEL:-$LR_ASD_MODEL_DIR/s3fd_270x480.onnx}"
onnx_dir="$HOME/SoloSeniorWatchRobot_build/onnxruntime-linux-x64-gpu-1.22.0/lib"

# whisper.cpp is built with a user-local CUDA toolkit on this workstation.
# Keep these libraries in the runtime search path so Whisper can load its
# CUDA/cuBLAS backend without requiring a system-wide CUDA installation.
cuda_whisper_lib="$HOME/.local/cuda-12.8/usr/local/cuda-12.8/targets/x86_64-linux/lib"

# This workstation gets CUDA user-space libraries from the LR-ASD Python
# environment. A robot with a system CUDA/cuDNN installation does not need
# these extra directories.
nvidia_root="$project_root/.venv/lib/python3.12/site-packages/nvidia"
cuda_paths=""
if [[ -d "$nvidia_root" ]]; then
    while IFS= read -r directory; do cuda_paths="${cuda_paths}${directory}:"; done \
        < <(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib)
fi
export LD_LIBRARY_PATH="${cuda_whisper_lib}:${cuda_paths}${onnx_dir}:${LD_LIBRARY_PATH:-}"

# Prefer the current local AnythingLLM API key over an old key saved in a
# robot JSON profile. This keeps the robot client synchronized with the
# local AnythingLLM installation.
anythingllm_db="$HOME/.config/anythingllm-desktop/storage/anythingllm.db"
if [[ -f "$anythingllm_db" ]]; then
    read -r local_anythingllm_key < <(python3 - "$anythingllm_db" <<'PY'
import sqlite3
import sys

db_path = sys.argv[1]
with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
    row = connection.execute(
        "select secret from api_keys order by id desc limit 1"
    ).fetchone()
    if row and row[0]:
        print(row[0])
PY
    )
    if [[ -n "${local_anythingllm_key:-}" ]]; then
        export ANYTHINGLLM_API_KEY="$local_anythingllm_key"
    fi
fi

if [[ $# -eq 0 ]]; then
    set -- "$(dirname "$0")/build/SoloSeniorWatchRobot"
fi
exec "$@"
