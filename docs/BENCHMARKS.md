# Python and C++ comparison

## Matched pipeline

Both implementations used the official LR-ASD and S3FD weights, 112x112
grayscale faces, 16 kHz mono audio, 13 MFCC coefficients, four MFCC rows per 25
FPS frame, a rolling 25-frame window, and inference every five frames.

Python uses PyTorch/OpenCV/CUDA. C++ uses LibTorch/OpenCV/CUDA. A paced test
simulates a 25 FPS source; an unpaced test measures processing capacity.

## Numerical parity

| Metric | Track 0 | Track 1 |
|---|---:|---:|
| Speaking-decision agreement | 100% | 100% |
| Mean absolute class-1 logit error | 0.00012 | 0.00000005 |

`python_speech_features` uses a rectangular analysis window by default. Matching
that detail removed the initial C++/Python difference.

## CUDA results

Hardware: NVIDIA GeForce RTX 4080 SUPER. Both implementations passed paced 25
FPS testing.

| Input | Frames | Python CUDA | C++ CUDA | C++ / Python |
|---|---:|---:|---:|---:|
| Wide/mobile recording | 534 | 44.10 FPS | 65.38 FPS | 1.48x |
| 1080p two-person recording | 802 | 36.60 FPS | 56.77 FPS | 1.55x |

Paced results were 25.0 FPS for both. Capacity is more informative because pacing
intentionally makes successful implementations finish at the same speed.

## CPU diagnostic

C++ CPU achieved 8.27 FPS on the wide recording. S3FD on every frame—not LR-ASD
itself—was the bottleneck. CUDA raised the same pipeline to 65.38 FPS. Therefore,
CPU C++ versus CUDA Python is not a fair language comparison.

## Limitations

- Results apply to these inputs and this machine, not every deployment.
- Tests used paced recorded files, not a live camera and microphone.
- Live deployment must test buffering, clock drift, dropped frames, long runs,
  and robot middleware.
- S3FD TorchScript exports are input-shape-specific. The exporter includes the
  two tested detector shapes at scale 0.25.
