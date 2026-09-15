#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$experiment_dir/../.." && pwd)"
python_bin="${PYTHON:-$project_dir/.venv/bin/python}"
site_dir="$("$python_bin" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent.parent)')"
lib_paths="$experiment_dir/third_party/python/onnxruntime/capi:$site_dir/torch/lib"
for folder in "$site_dir"/nvidia/*/lib; do lib_paths="$lib_paths:$folder"; done
export LD_LIBRARY_PATH="$lib_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$experiment_dir/build/lr_asd_scrfd" "$@"
