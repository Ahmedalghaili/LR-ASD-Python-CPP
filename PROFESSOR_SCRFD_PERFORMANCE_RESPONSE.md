# SCRFD performance and real-time evaluation

Prepared: 2026-09-14

## Short answer

Yes. Replacing S3FD with SCRFD makes the face-detection stage faster and much
smaller in the tested C++ pipeline. The recommended configuration is SCRFD-2.5G
at 480×288.

On an NVIDIA RTX 4080 SUPER, the complete pipeline processed approximately
78–95 frames per second, compared with 63–81 FPS for S3FD. The target pipeline
rate is 25 FPS, so the measured SCRFD pipeline had approximately 3× real-time
headroom on this workstation.

The robot still needs an on-device benchmark. The current results prove the
optimization on the workstation, but they do not prove the final speed on the
robot's CPU/GPU.

## Experimental setup

The comparison uses the C++ LR-ASD pipeline with:

- face detection;
- face-box decoding and NMS;
- IoU tracking;
- 112×112 grayscale face cropping;
- MFCC preparation and LR-ASD inference;
- overlays, CSV output, and video encoding.

The benchmark uses CUDA, FP32 models, ten warmup calls, synchronized GPU
timing, and three full-video repeats per configuration. It excludes model
loading, MFCC preparation before the loop, and warmup. The pipeline processes
the source timeline at 25 FPS without artificial pacing.

Hardware and software details are recorded in:

```text
experiments/scrfd_cpp/environment.json
experiments/scrfd_cpp/results/cuda_480x288_verified/config.json
experiments/scrfd_cpp/results/cuda_480x288_verified/runs.json
```

## S3FD versus SCRFD

The values below are medians from three full-video repeats.

| Recording | S3FD pipeline | SCRFD-2.5G pipeline | Speed increase | S3FD detector | SCRFD detector | Detector reduction |
|---|---:|---:|---:|---:|---:|---:|
| Wide | 81.15 FPS | 94.55 FPS | 16.5% | 3.99 ms | 2.08 ms | 47.8% |
| Two-person | 63.05 FPS | 77.55 FPS | 23.0% | 5.03 ms | 2.20 ms | 56.2% |

The result is a faster complete pipeline, not only a faster isolated detector.
The full-pipeline improvement is smaller than the detector improvement because
video decoding, memory transfers, face cropping, LR-ASD, tracking, and output
encoding also consume time.

## Real-time margin

At 25 FPS, each frame has a 40 ms processing budget.

| Configuration | Average time per processed frame | Margin against 25 FPS |
|---|---:|---:|
| S3FD, two-person recording | 15.86 ms | 2.52× faster than required |
| SCRFD-2.5G, two-person recording | 12.90 ms | 3.10× faster than required |
| SCRFD-2.5G, wide recording | 10.58 ms | 3.78× faster than required |

The measured SCRFD frame-time 95th percentile on the more demanding
two-person recording was 14.60 ms, equivalent to approximately 68.5 FPS. This
still leaves substantial margin over 25 FPS on the test workstation.

It also exceeds a 30 FPS target in this workstation experiment, although the
robot should be evaluated against its actual camera rate and timing behavior.

## Other optimization approaches explored

### 1. Smaller SCRFD model

SCRFD-2.5G was compared with SCRFD-10G at the same input sizes.

At 480×288:

| Model | Wide | Two-person |
|---|---:|---:|
| SCRFD-2.5G | 94.55 FPS | 77.55 FPS |
| SCRFD-10G | 92.09 FPS | 76.05 FPS |

SCRFD-2.5G is slightly faster and is approximately 3.3 MB, while the 10G
model is approximately 17 MB. SCRFD-2.5G is therefore the better deployment
choice unless a labeled evaluation shows a meaningful accuracy advantage for
10G.

### 2. Lower detector input resolution

The 480×288 configuration was compared with 640×640.

| Input/model | Wide | Two-person |
|---|---:|---:|
| SCRFD-2.5G, 480×288 | 94.55 FPS | 77.55 FPS |
| SCRFD-2.5G, 640×640 | 86.19 FPS | 69.11 FPS |

480×288 reduces the detector input area by approximately 66% relative to
640×640 and improves full-pipeline throughput by about 8–12% in this test.
It also matches the 16:9 camera content better than a square input. This is the
recommended starting point for the robot.

### 3. Existing temporal optimizations

The pipeline already limits LR-ASD inference to every five processed frames
while retaining a 25-frame face history. This reduces repeated ASD work while
preserving the temporal input expected by LR-ASD.

The detector itself currently runs for every processed frame in the experimental
benchmark. A possible next optimization is detector interval scheduling:
detect every 2–5 frames and propagate boxes with tracking between detections.
This must be evaluated carefully because missed or stale boxes can affect face
crops and active-speaker decisions.

### 4. Runtime optimization already used

SCRFD runs through ONNX Runtime with CUDA, graph optimization enabled, one
intra-op CPU thread, synchronized measurements, and a warmup phase. These
settings avoid measuring first-run initialization as steady-state throughput.

FP16, TensorRT execution, and detector-interval scheduling have not yet been
validated on the robot. They are the next optimization candidates after the
basic SCRFD integration is working.

## Detection consistency and limitations

The C++ SCRFD output was checked against an independent Python SCRFD reference:

- 32 sampled model/frame checks passed at 480×288;
- maximum box-coordinate difference was below 0.028 pixels;
- track counts were stable on the tested recordings;
- speaking-decision agreement with S3FD was 98.23–98.44% for SCRFD-2.5G.

The personal recordings do not have manual face or speaking labels. Therefore,
agreement with S3FD is not proof that either detector is more accurate. A
labeled face-detection test set is still required to claim an accuracy
improvement.

## Robot status

SCRFD is implemented and tested in the experimental executable, but the
production robot server still loads S3FD. The production worker currently
references `s3fd_270x480.onnx` here:

```text
SoloSeniorWatchRobot-main/Server/LRASDWorker.cpp:110
```

The experimental implementation is here:

```text
experiments/scrfd_cpp/src/main.cpp
experiments/scrfd_cpp/src/scrfd.hpp
experiments/scrfd_cpp/build/lr_asd_scrfd
```

The recommended model is:

```text
experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx
```

SCRFD must not be deployed by simply renaming it to the S3FD model filename.
Its preprocessing, output tensors, decoding, and NMS are different. The
SCRFD implementation must be integrated into the production worker or selected
as a separate detector backend.

## Next tests on the robot

1. Build SCRFD and ONNX Runtime for the robot's architecture.
2. Run the same camera recording through S3FD and SCRFD-2.5G.
3. Measure steady-state FPS, average latency, 95th-percentile latency, RAM,
   GPU memory, and CPU/GPU utilization.
4. Verify that the camera rate is sustained for at least 5–10 minutes.
5. Check box stability, track resets, audio/video synchronization, and active
   speaker decisions.
6. Compare SCRFD-2.5G at 480×288 with SCRFD-10G and 640×640 only if the first
   configuration does not meet the robot's accuracy requirements.
7. After the baseline is stable, evaluate FP16/TensorRT and detector interval
   scheduling.

## Reproducibility commands

Run the existing C++ comparison benchmark from the repository root:

```bash
cd /home/ahmed/Desktop/Ahmed/experiments/LR-ASD-main
source .venv/bin/activate

python experiments/scrfd_cpp/scripts/benchmark.py \
  --repeats 3 \
  --width 480 \
  --height 288 \
  --output experiments/scrfd_cpp/results/new_480x288
```

Run the comparison unit tests:

```bash
cd /home/ahmed/Desktop/Ahmed/experiments/LR-ASD-main/experiments/scrfd_cpp
source ../../.venv/bin/activate
python -m unittest discover -s scripts -p 'test_comparison.py' -v
```

Full source report:

```text
experiments/scrfd_cpp/RESULTS.md
```

Robot handoff notes:

```text
SCRFD_ROBOT_UPDATE.md
```

## Professor-ready conclusion

> SCRFD runs faster than S3FD in our C++ LR-ASD pipeline. Using SCRFD-2.5G at
> 480×288 reduced detector latency by approximately 48–56% and increased total
> throughput by 16.5–23.0% on an RTX 4080 SUPER. The complete pipeline reached
> 77.55 FPS in the more demanding two-person recording and 94.55 FPS in the
> wide recording, compared with a 25 FPS real-time target. This gives roughly
> 3× real-time headroom on the workstation. We also evaluated model size and
> input resolution: SCRFD-2.5G was preferable to SCRFD-10G, and 480×288 was
> faster than 640×640. The next step is to integrate SCRFD into the production
> robot worker and repeat the measurements on the robot hardware, followed by
> FP16/TensorRT and detector-interval experiments. Because the current videos
> lack manual labels, the reported detection agreement is not yet an accuracy
> claim.
