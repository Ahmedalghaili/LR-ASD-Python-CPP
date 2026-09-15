# SCRFD versus S3FD: C++ results

SCRFD-2.5G at **480×288** is the recommended starting configuration for further evaluation on these recordings. It reduces detector latency by roughly half and improves total video-loop throughput, while changing some detections and speaking decisions.

Measured on an NVIDIA RTX 4080 SUPER, CUDA, three full-video repeats per configuration. Values below are medians. Both original recordings were tested (534 and 802 processed frames). The original `cpp/` implementation and checkpoints are unchanged.

## Main comparison: 480×288 SCRFD

| Recording | S3FD FPS | SCRFD-2.5G FPS | FPS gain | S3FD detector ms | SCRFD detector ms | Detector time reduction |
|---|---:|---:|---:|---:|---:|---:|
| wide | 81.15 | 94.55 | 16.5% | 3.99 | 2.08 | 47.8% |
| two_person | 63.05 | 77.55 | 23.0% | 5.03 | 2.20 | 56.2% |

Halving detector time does not double total FPS because decoding, cropping, LR-ASD, transfers and output encoding still take time. Total FPS here excludes model loading, audio-feature preparation and warmup.

## All tested configurations

| Input | Recording | Detector | Median FPS | Min–max FPS | Speedup over same-run S3FD | Detector ms |
|---|---|---|---:|---:|---:|---:|
| 480×288 | wide | s3fd | 81.15 | 80.61–81.32 | 1.00× | 3.99 |
| 480×288 | wide | scrfd_2.5g | 94.55 | 94.50–95.00 | 1.17× | 2.08 |
| 480×288 | wide | scrfd_10g | 92.09 | 91.79–93.07 | 1.13× | 2.28 |
| 480×288 | two_person | s3fd | 63.05 | 62.98–64.35 | 1.00× | 5.03 |
| 480×288 | two_person | scrfd_2.5g | 77.55 | 77.43–78.13 | 1.23× | 2.20 |
| 480×288 | two_person | scrfd_10g | 76.05 | 75.31–76.39 | 1.21× | 2.42 |
| 640×640 | wide | s3fd | 80.46 | 80.43–80.68 | 1.00× | 4.05 |
| 640×640 | wide | scrfd_2.5g | 86.19 | 84.62–87.62 | 1.07× | 3.24 |
| 640×640 | wide | scrfd_10g | 77.14 | 75.97–79.79 | 0.96× | 4.59 |
| 640×640 | two_person | s3fd | 63.82 | 62.98–64.72 | 1.00× | 5.00 |
| 640×640 | two_person | scrfd_2.5g | 69.11 | 66.34–74.36 | 1.08× | 3.81 |
| 640×640 | two_person | scrfd_10g | 64.73 | 64.73–65.18 | 1.01× | 4.97 |

S3FD uses its original 0.25 image scale in both sets. SCRFD uses top-left letterboxing at the listed input size. Thresholds follow each implementation: S3FD confidence 0.9/NMS 0.1, SCRFD confidence 0.5/NMS 0.4. These compare complete deployment configurations, including different runtimes and resolution; they are not isolated architecture comparisons.

## Detection and speaking differences at 480×288

| Recording | Model | Matched observations | Unmatched S3FD / SCRFD | Speaking agreement | Total tracks S3FD / SCRFD |
|---|---|---:|---:|---:|---:|
| wide | scrfd_2.5g | 566 | 2 / 0 | 98.23% | 7 / 7 |
| wide | scrfd_10g | 568 | 0 / 57 | 98.24% | 7 / 8 |
| two_person | scrfd_2.5g | 1604 | 0 / 0 | 98.44% | 2 / 2 |
| two_person | scrfd_10g | 1604 | 0 / 0 | 97.51% | 2 / 2 |

**Agreement is not accuracy.** Detections are matched by per-frame IoU ≥ 0.5, independent of track IDs. Speaking agreement covers only matched observations and includes initial default scores. The videos have no manual ground truth, so unmatched detections cannot be classified as errors and no ASD mAP or face recall is claimed. Inspect the videos and evaluate labeled data before deployment.

## Validation

- `cuda_480x288_verified`: all 32 sampled model/frame checks matched the official Python reference's box counts; maximum coordinate difference 0.0280 pixels, below the 0.1-pixel tolerance. Samples span both complete videos and use sequential decoding to handle variable frame timing.
- `cuda_640x640`: all 32 sampled model/frame checks matched the official Python reference's box counts; maximum coordinate difference 0.0139 pixels, below the 0.1-pixel tolerance. Samples span both complete videos and use sequential decoding to handle variable frame timing.
- Original C++ versus experimental S3FD: 1,604 two-person observations had identical boxes, track IDs and speaking decisions. Maximum LR-ASD score difference was 0.0001.
- Comparison tests cover reordered track IDs, unmatched detections, track changes and no-match cases. Model checksums and ONNX validity were checked; dynamic-shape metadata changes preserve all network operations and weights.

## Open the results

| Recording | S3FD | SCRFD-2.5G 480×288 | SCRFD-10G 480×288 |
|---|---|---|---|
| wide | [Play](results/cuda_480x288_verified/wide_s3fd_r1.mp4) | [Play](results/cuda_480x288_verified/wide_scrfd_2.5g_r1.mp4) | [Play](results/cuda_480x288_verified/wide_scrfd_10g_r1.mp4) |
| two_person | [Play](results/cuda_480x288_verified/two_person_s3fd_r1.mp4) | [Play](results/cuda_480x288_verified/two_person_scrfd_2.5g_r1.mp4) | [Play](results/cuda_480x288_verified/two_person_scrfd_10g_r1.mp4) |

- [480×288 detailed report](results/cuda_480x288_verified/REPORT.md)
- [640×640 detailed report](results/cuda_640x640/REPORT.md)
- [Build and run instructions](README.md)
- [Software versions](environment.json)

Each final results folder contains the exact commands and hashes (`config.json`, `runs.json`), aggregates (`summary.json`), reference checks (`validation.json`), per-frame timing CSVs, detection/speaking CSVs, stderr logs and annotated videos. Generated models, dependencies, binaries and raw result folders are excluded from Git; this summary is retained.

Measurements use FP32 models with runtime-default math settings, ten warmup calls, synchronized CUDA timing and sequential GPU runs with rotated order. They include video-loop I/O and exclude startup and MFCC preparation. Existing track-gap timing behavior is inherited. These are file-processing capacity measurements, not a live camera/microphone latency test.

Model source URLs, checksums, upstream references and research-use terms are documented in [README.md](README.md).
