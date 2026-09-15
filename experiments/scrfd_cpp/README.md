# SCRFD detector experiment for LR-ASD C++

This folder compares the existing S3FD detector with SCRFD-2.5G and SCRFD-10G,
using the existing LR-ASD checkpoint on the RTX 4080 SUPER. Start with
[RESULTS.md](RESULTS.md) for measured results and limitations.

The original `cpp/` implementation is unchanged. The experimental executable
reuses its S3FD, MFCC and face-cropping helpers and preserves its rolling
25-frame window, inference every five frames, and IoU tracking behavior.
SCRFD uses ONNX Runtime CUDA; LR-ASD and S3FD use LibTorch CUDA. No retraining
is needed to run this experiment.

## Folder layout

```text
experiments/scrfd_cpp/
├── README.md                 instructions
├── RESULTS.md                measured comparison and recommendation
├── build.sh                  build the separate C++ executable
├── run.sh                    set runtime library paths and run it
├── src/main.cpp              selectable detector and instrumented pipeline
├── src/scrfd.hpp              native C++ SCRFD preprocessing/decoding/NMS
├── scripts/setup.py          pinned downloads and model checksums
├── scripts/benchmark.py      repeated sequential runs and agreement report
├── scripts/validate.py       compare boxes with the official Python reference
├── scripts/summarize.py      regenerate the combined report for the saved runs
├── scripts/test_comparison.py matching/track comparison tests
├── build/                    generated executable (ignored by Git)
├── models/                   downloaded ONNX weights (ignored)
├── third_party/              local ONNX Runtime and reference code (ignored)
└── results/<run>/            commands, hashes, JSON, CSV, logs, annotated MP4s
```

## Build

Run these commands from the repository root:

```bash
.venv/bin/python experiments/scrfd_cpp/scripts/setup.py
bash experiments/scrfd_cpp/build.sh
```

Requirements: the project's existing `.venv` with CUDA PyTorch, NumPy, ONNX,
OpenCV Python and FFmpeg; g++; native OpenCV in `/usr/local`; CUDA/cuDNN runtime
libraries supplied by the existing environment. The setup installs ONNX Runtime
GPU **1.22.0** into this experiment's `third_party/python/`, leaving the project's
installed Python packages intact. The build discovers PyTorch paths and its C++
ABI. Set `PYTHON=/path/to/python` for the build/run scripts if needed.

The existing `cpp/models/lr_asd.pt` and S3FD TorchScript exports are reused. On a
fresh checkout, generate them first with `.venv/bin/python cpp/export_torchscript.py`
using the original project's setup.

## Run SCRFD on a video

```bash
mkdir -p experiments/scrfd_cpp/results/manual
ffmpeg -y -i VID_20260824_145739.mp4 -ac 1 -ar 16000 -vn -c:a pcm_s16le \
  experiments/scrfd_cpp/results/manual/audio.wav

bash experiments/scrfd_cpp/run.sh \
  --detector scrfd \
  --model experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx \
  --asd cpp/models/lr_asd.pt \
  --video VID_20260824_145739.mp4 \
  --audio experiments/scrfd_cpp/results/manual/audio.wav \
  --output experiments/scrfd_cpp/results/manual/scrfd.mp4 \
  --width 480 --height 288 --device cuda
```

For SCRFD-10G, substitute `scrfd_10g_bnkps.dynamic.onnx`. For S3FD, use
`--detector s3fd --model cpp/models/s3fd_270x480.pt` for the 1080p recording.
S3FD uses the original preprocessing and thresholds; `--width`, `--height`,
`--score` and `--nms` apply only to SCRFD. SCRFD dimensions must be multiples
of 32; defaults are 640×640, score 0.5, NMS 0.4. Keypoints are unused.

The command produces an annotated MP4, `<output>.csv` with tracks and speaking
scores, `<output>.timing.csv` with per-frame timings, and a JSON metric line on
stdout. It overwrites files at the chosen output path. `--max-frames N` limits a
quick check; default 0 processes the full video. `--warmup N` defaults to 10.
Runs are unpaced; outputs retain the original pipeline's 25 FPS timeline.
Audio must correspond to the same video from its beginning.

## Repeat the benchmark

```bash
.venv/bin/python experiments/scrfd_cpp/scripts/benchmark.py \
  --repeats 3 --width 480 --height 288

.venv/bin/python experiments/scrfd_cpp/scripts/benchmark.py \
  --repeats 3 --width 640 --height 640
```

Each invocation creates a new dated results directory and runs the three
configurations sequentially on both recordings. It rotates detector order
between repeats and records medians, min/max FPS, and detector latency.
Use `--output experiments/scrfd_cpp/results/NEW_NAME` for a specific **new** folder;
existing result folders are never overwritten by a fresh benchmark. To finish
an interrupted benchmark, repeat its command with the same `--output` and add
`--resume`. The runner verifies configuration and hashes, preserves completed
runs, and reruns only unfinished configurations.
The two default input paths are in `VIDEOS` in `scripts/benchmark.py`.

Validate C++ detection boxes against the pinned InsightFace reference:

```bash
.venv/bin/python experiments/scrfd_cpp/scripts/validate.py \
  experiments/scrfd_cpp/results/cuda_480x288_verified
```

Validation uses Python CPU ONNX Runtime as an independent implementation and
allows <0.1 pixel coordinate difference against C++ CUDA. It samples frames
from each video/model, compares box counts and matches boxes by overlap.
Frames are decoded sequentially, matching the C++ pipeline even for variable
frame timing. Run `.venv/bin/python experiments/scrfd_cpp/scripts/test_comparison.py`
to check the report matching logic.

## Interpreting results

FPS measures the video loop, including video reading/writing and transfers;
model loading, MFCC preparation and warmup are excluded. GPU work is synchronized
for detector/ASD timings. Compare fresh S3FD results from the same run, rather
than historical FPS obtained with different warmup and timing boundaries.

The smaller SCRFD input approximately matches the existing S3FD pixel scale.
The 640×640 configuration tests the commonly used larger detector input.
These are complete deployment comparisons with different runtimes and
thresholds, not a controlled FLOP-only comparison.

Speaking agreement with S3FD is **not accuracy**. These two recordings have no
manual ground truth. Unmatched boxes are not automatically missed faces or
false positives. Inspect the annotated outputs and use labeled evaluation
before choosing a detector for deployment. Existing temporal behavior on
missed detections is inherited from the original pipeline.

## Training versus inference

This experiment replaces the face detector used during inference. LR-ASD's
architecture and checkpoint are unchanged, and pretrained SCRFD weights are
used. Training LR-ASD is still handled by the original Python `train.py`;
the C++ executable is for inference.

AVA training requires extracted audio clips and face crops under
`AVADataPath/clips_audios/{train,val}` and `AVADataPath/clips_videos/{train,val}`.
Downloading the videos alone does not create these. The original AVA crop
preparation uses annotation boxes, so it does not require running SCRFD or S3FD.

## Sources and model provenance

- [Official SCRFD project](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [Reference inference code, pinned commit](https://github.com/deepinsight/insightface/blob/41bf106fa0f9a3c998dd717f2abb5f3c2fada6da/detection/scrfd/tools/scrfd.py)
- [ONNX Runtime CUDA documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [InsightFace model usage terms](https://github.com/deepinsight/insightface#license)

The official OneDrive model link returned HTTP 403 during setup. The experiment
therefore downloads ONNX files from pinned Hugging Face mirrors, not from a
verified official distribution. Exact URLs and expected SHA-256 digests are in
`scripts/setup.py`; checksums ensure reproducibility, not publisher authenticity.
These are the keypoint variants `scrfd_2.5g_bnkps` and `scrfd_10g_bnkps`.
InsightFace pretrained weights are supplied for non-commercial research;
mirror metadata does not override the upstream model terms. Original LR-ASD
code/checkpoint attribution remains in the repository root README.

Setup retains the checksum-verified downloads and creates `.dynamic.onnx` files
with corrected dynamic output-shape metadata. Their graph operations and weights
are identical to the downloads; this avoids stale 640×640 shape warnings at
smaller input resolutions. The benchmark uses these derived files.
