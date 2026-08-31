# LR-ASD Python and C++ CUDA Evaluation

This repository extends the official
[LR-ASD implementation](https://github.com/Junhua-Liao/LR-ASD) with a native C++
pipeline and a reproducible Python-versus-C++ CUDA comparison.

LR-ASD combines synchronized face motion and audio features to determine which
visible person is speaking.

> Datasets, personal videos, generated outputs, compiled binaries, and downloaded
> SDKs are intentionally excluded.

## Included

- Official LR-ASD Python model and evaluation code
- Official AVA and TalkSet-finetuned checkpoints
- Official S3FD face detector with automatic weight download
- Near-real-time Python video pipeline
- C++17 LibTorch/OpenCV implementation
- Native C++ MFCC matching Python settings
- C++ CPU and CUDA build paths
- TorchScript export scripts
- Reproducible benchmark methodology

## Pipeline

```text
Video -> S3FD -> IoU tracking -> 112x112 grayscale faces ─┐
                                                          ├-> LR-ASD -> speaking score
Audio -> 16 kHz mono -> 13-D MFCC -> A/V synchronization ─┘
```

Both implementations use a rolling 25-frame window and update predictions every
five frames.

## Python versus C++ result

Both CUDA implementations sustained paced 25 FPS operation. Unpaced capacity on
the same RTX 4080 SUPER and exact same recordings was:

| Input | Python CUDA | C++ CUDA | C++ speedup |
|---|---:|---:|---:|
| Wide/mobile recording | 44.10 FPS | 65.38 FPS | 1.48x |
| 1080p two-person recording | 36.60 FPS | 56.77 FPS | 1.55x |

C++ and Python speaking decisions agreed on 100% of 1,604 parity frames. See
[the benchmark report](docs/BENCHMARKS.md) for methodology and limitations.

## Requirements

- Ubuntu 24.04 or comparable Linux
- Python 3.12
- FFmpeg
- OpenCV development libraries
- CMake and a C++17 compiler
- NVIDIA GPU and compatible driver for CUDA

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## Python

Official demo:

```bash
python Columbia_test.py --videoName example --videoFolder demo
```

Use `demo/example.mp4`. The annotated output is written below
`demo/example/pyavi/`.

Rolling pipeline:

```bash
python realtime_video.py input.mp4 --output python_output.mp4 --no-pace
```

Remove `--no-pace` for paced 25 FPS playback.

## C++

Export the official models:

```bash
python cpp/export_torchscript.py
```

CPU build:

```bash
cmake -S cpp -B cpp/build_cpu \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch;/usr/local/lib/cmake/opencv4" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build_cpu -j2
```

CUDA build using CUDA-enabled PyTorch libraries in `.venv`:

```bash
pip install nvidia-cuda-nvcc-cu12==12.8.93
bash cpp/build_cuda.sh
```

Prepare audio and run:

```bash
ffmpeg -y -i input.mp4 -ac 1 -ar 16000 -vn -c:a pcm_s16le audio.wav

LR_ASD_DEVICE=cuda cpp/build_cuda/lr_asd --video \
  cpp/models/lr_asd.pt cpp/models/s3fd_270x480.pt \
  input.mp4 audio.wav cpp_output.mp4
```

Add `LR_ASD_PACE=1` for a paced 25 FPS test. The S3FD model name must match
`round(height * 0.25)` x `round(width * 0.25)`. Add new shapes in
`cpp/export_torchscript.py` when testing a different resolution.

## Datasets

No dataset is uploaded. See [dataset instructions](docs/DATASETS.md) for official
download links, storage warnings, directory structure, and evaluation commands.

## Repository hygiene

`.gitignore` excludes AVA/Columbia data, videos, audio, images, predictions,
virtual environments, LibTorch downloads, TorchScript models, and binaries.

Before committing, check for accidental large files:

```bash
git status --short
find . -type f -size +50M -not -path './.git/*'
```

## Attribution

The official LR-ASD project reports 94.45% mAP on AVA validation. Cite the
original authors when using this work:

```bibtex
@article{liao2025lrasd,
  title={LR-ASD: Lightweight and Robust Network for Active Speaker Detection},
  author={Liao, Junhua and Duan, Haihan and Feng, Kanghui and Zhao, Wanbing and
          Yang, Yanbing and Chen, Liangyin and Chen, Yanru},
  journal={International Journal of Computer Vision},
  year={2025},
  publisher={Springer}
}
```

Upstream: <https://github.com/Junhua-Liao/LR-ASD>
