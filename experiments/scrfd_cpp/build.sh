#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$experiment_dir/../.." && pwd)"
python_bin="${PYTHON:-$project_dir/.venv/bin/python}"
torch_dir="$("$python_bin" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')"
site_dir="$(dirname "$torch_dir")"
ort_dir="$experiment_dir/third_party/python/onnxruntime/capi"
abi="$("$python_bin" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
mkdir -p "$experiment_dir/build"
g++ -O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI="$abi" \
 "$experiment_dir/src/main.cpp" -o "$experiment_dir/build/lr_asd_scrfd" \
 -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
 -I/usr/local/include/opencv4 -I"$experiment_dir/third_party/include" \
 -L"$torch_dir/lib" -L"$site_dir/nvidia/cuda_runtime/lib" -L/usr/local/lib -L"$ort_dir" \
 -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 \
 -l:libcudart.so.12 -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs \
 -l:libonnxruntime.so.1.22.0 \
 -Wl,-rpath,"$torch_dir/lib:$site_dir/nvidia/cuda_runtime/lib:/usr/local/lib:$ort_dir" -pthread
