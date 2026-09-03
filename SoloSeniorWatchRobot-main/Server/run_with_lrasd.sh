#!/usr/bin/env bash
set -euo pipefail

export LR_ASD_MODEL_DIR="${LR_ASD_MODEL_DIR:-$HOME/SoloSeniorWatchRobot_build/lr-asd}"
onnx_dir="$HOME/SoloSeniorWatchRobot_build/onnxruntime-linux-x64-gpu-1.22.0/lib"

# This workstation gets CUDA user-space libraries from the LR-ASD Python
# environment. A robot with a system CUDA/cuDNN installation does not need
# these extra directories.
project_root="$(cd "$(dirname "$0")/../.." && pwd)"
nvidia_root="$project_root/.venv/lib/python3.12/site-packages/nvidia"
cuda_paths=""
if [[ -d "$nvidia_root" ]]; then
    while IFS= read -r directory; do cuda_paths="${cuda_paths}${directory}:"; done \
        < <(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib)
fi
export LD_LIBRARY_PATH="${cuda_paths}${onnx_dir}:${LD_LIBRARY_PATH:-}"

if [[ $# -eq 0 ]]; then
    set -- "$(dirname "$0")/build/SoloSeniorWatchRobot"
fi
exec "$@"
