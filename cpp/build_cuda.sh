#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
venv_dir="$project_dir/.venv"
torch_dir="$venv_dir/lib/python3.12/site-packages/torch"
cuda_runtime_dir="$venv_dir/lib/python3.12/site-packages/nvidia/cuda_runtime"
mkdir -p "$project_dir/cpp/build_cuda"
g++ -O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
  "$project_dir/cpp/src/main.cpp" -o "$project_dir/cpp/build_cuda/lr_asd" \
  -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
  -I/usr/local/include/opencv4 \
  -L"$torch_dir/lib" -L"$cuda_runtime_dir/lib" -L/usr/local/lib \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 \
  -l:libcudart.so.12 -lopencv_core -lopencv_imgproc -lopencv_videoio \
  -lopencv_imgcodecs \
  -Wl,-rpath,"$torch_dir/lib:$cuda_runtime_dir/lib:/usr/local/lib" -pthread
echo "$project_dir/cpp/build_cuda/lr_asd"
