# SCRFD robot update

Updated: 2026-09-12

## Purpose

This document records the S3FD-to-SCRFD experiment for the C++ LR-ASD
pipeline. It is a handoff for testing the detector on the robot later.

SCRFD-2.5G is the recommended detector configuration. In the workstation
benchmark it was substantially smaller and faster than the existing S3FD
detector:

| Detector | Model size | Detector latency | Full pipeline |
|---|---:|---:|---:|
| S3FD | about 90 MB | 4–5 ms | 63–81 FPS |
| SCRFD-2.5G | about 3.3 MB | about 2.2 ms | 78–95 FPS |

The benchmark was run on an NVIDIA RTX 4080 SUPER. These numbers must be
rechecked on the robot's actual CPU/GPU.

## Current implementation status

SCRFD is already implemented and tested in the separate experimental C++
executable. It is selectable with `--detector scrfd` and uses ONNX Runtime
CUDA.

The production robot server now supports both detectors. It defaults to S3FD
for rollback compatibility and can select SCRFD with `LR_ASD_DETECTOR=scrfd`.
Do not replace the S3FD file by renaming the SCRFD file: the output format,
preprocessing, decoding, and NMS are different.

## Important paths

Repository root:

```text
/home/ahmed/Desktop/Ahmed/experiments/LR-ASD-main
```

SCRFD C++ experiment:

```text
experiments/scrfd_cpp/src/main.cpp
experiments/scrfd_cpp/src/scrfd.hpp
experiments/scrfd_cpp/build/lr_asd_scrfd
experiments/scrfd_cpp/build.sh
experiments/scrfd_cpp/run.sh
experiments/scrfd_cpp/scripts/setup.py
experiments/scrfd_cpp/scripts/test_comparison.py
experiments/scrfd_cpp/scripts/validate.py
experiments/scrfd_cpp/README.md
experiments/scrfd_cpp/RESULTS.md
```

SCRFD models:

```text
experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx
experiments/scrfd_cpp/models/scrfd_10g_bnkps.dynamic.onnx
```

Use this model first:

```text
experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx
```

Existing S3FD comparison models:

```text
cpp/models/s3fd_206x466.pt
cpp/models/s3fd_270x480.pt
```

Robot production implementation:

```text
SoloSeniorWatchRobot-main/Server/LRASDWorker.cpp
SoloSeniorWatchRobot-main/Server/LRASDWorker.hpp
SoloSeniorWatchRobot-main/Server/CMakeLists.txt
SoloSeniorWatchRobot-main/Server/LR-ASD-INTEGRATION.md
SoloSeniorWatchRobot-main/Server/SCRFDDetector.hpp
```

The original production detector path is:

```text
<LR_ASD_MODEL_DIR>/s3fd_270x480.onnx
```

The SCRFD production detector path is:

```text
<LR_ASD_MODEL_DIR>/scrfd_2.5g_bnkps.dynamic.onnx
```

The production worker selects the detector in:

```text
SoloSeniorWatchRobot-main/Server/LRASDWorker.cpp:110
```

The switch implementation is in:

```text
SoloSeniorWatchRobot-main/Server/SCRFDDetector.hpp
SoloSeniorWatchRobot-main/Server/run_with_lrasd.sh
```

## New LR-ASD checkpoint

The newly trained LR-ASD checkpoint used in the C++ smoke test is:

```text
experiments/trained_model_test/results/20260912_003121/models/lr_asd_epoch32.pt
```

It is about 3.5 MB. This is a newly trained checkpoint of the same LR-ASD
architecture; it is not a replacement detector architecture.

Evaluation report:

```text
experiments/trained_model_test/RESULTS.md
experiments/trained_model_test/results/20260912_003121/REPORT.md
```

The checkpoint loaded and ran successfully in C++, but its AVA validation AP
was slightly below the original checkpoint:

```text
original: 94.4850% AP
epoch 32: 94.3302% AP
```

For the first robot detector test, keep the original LR-ASD checkpoint and
change only S3FD to SCRFD. This isolates the detector speed change. Test the
new LR-ASD checkpoint separately afterward.

## Build the experimental C++ executable

Run from the repository root. The existing environment is used; no model
training is needed.

```bash
cd /home/ahmed/Desktop/Ahmed/experiments/LR-ASD-main
source .venv/bin/activate

python experiments/scrfd_cpp/scripts/setup.py
bash experiments/scrfd_cpp/build.sh
```

The build output is:

```text
experiments/scrfd_cpp/build/lr_asd_scrfd
```

If the robot is a different architecture or does not have the same CUDA,
cuDNN, LibTorch, OpenCV, and ONNX Runtime libraries, build the equivalent
dependencies on the robot rather than copying the x86 executable.

## Run a video smoke test

The SCRFD input dimensions below are important. They are width 480 by height
288, both multiples of 32.

```bash
cd /home/ahmed/Desktop/Ahmed/experiments/LR-ASD-main

bash experiments/scrfd_cpp/run.sh \
  --detector scrfd \
  --model experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx \
  --asd cpp/models/lr_asd.pt \
  --video 'WhatsApp Video 2026-08-30 at 9.46.47 PM.mp4' \
  --audio experiments/trained_model_test/results/20260912_003121/wide.wav \
  --output /tmp/scrfd_robot_check.mp4 \
  --width 480 \
  --height 288 \
  --device cuda \
  --max-frames 300 \
  --warmup 10
```

To test the newly trained LR-ASD checkpoint instead, change only the `--asd`
argument:

```bash
--asd experiments/trained_model_test/results/20260912_003121/models/lr_asd_epoch32.pt
```

The command creates these files beside the output video:

```text
/tmp/scrfd_robot_check.mp4
/tmp/scrfd_robot_check.mp4.csv
/tmp/scrfd_robot_check.mp4.timing.csv
```

The final JSON line reports `pipeline_fps`, `detector_mean_ms`,
`tracks_created`, `face_observations`, and `asd_calls`.

## Run the consistency tests

Comparison logic tests:

```bash
cd /home/ahmed/Desktop/Ahmed/experiments/LR-ASD-main/experiments/scrfd_cpp
source ../../.venv/bin/activate
python -m unittest discover -s scripts -p 'test_comparison.py' -v
```

Compare the saved C++ SCRFD boxes with the independent Python reference:

```bash
python scripts/validate.py results/cuda_480x288_verified
```

Expected result from the current workstation artifacts:

```text
32 sampled model/frame checks pass.
Maximum coordinate error: below 0.028 pixels.
```

## Production robot integration

The production `LRASDWorker` now includes SCRFD preprocessing, ONNX inference,
output decoding, NMS, and an environment-variable model switch.

Use the original S3FD backend:

```bash
LR_ASD_DETECTOR=s3fd \
  bash run_with_lrasd.sh build/SoloSeniorWatchRobot \
  --SettingFile json/ROGG16_SSWR.json
```

Use SCRFD-2.5G:

```bash
LR_ASD_DETECTOR=scrfd \
LR_ASD_SCRFD_MODEL=/path/to/scrfd_2.5g_bnkps.dynamic.onnx \
  bash run_with_lrasd.sh build/SoloSeniorWatchRobot \
  --SettingFile json/ROGG16_SSWR.json
```

If SCRFD is copied into the normal model directory, the path override is not
needed:

```bash
cp experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx \
  "$HOME/SoloSeniorWatchRobot_build/lr-asd/"
LR_ASD_DETECTOR=scrfd \
  bash run_with_lrasd.sh build/SoloSeniorWatchRobot \
  --SettingFile json/ROGG16_SSWR.json
```

The implementation keeps the existing 112×112 grayscale crops, MFCC
processing, tracking, and LR-ASD inputs unchanged. `LR_ASD_DETECTOR` selects
`s3fd` or `scrfd`; S3FD remains the default for rollback. The optional
`LR_ASD_SCRFD_MODEL` and `LR_ASD_S3FD_MODEL` variables override model paths.

SCRFD uses top-left letterboxing at 480×288 with score threshold 0.5 and NMS
threshold 0.4. S3FD uses different preprocessing and thresholds.

SCRFD uses top-left letterboxing and a score threshold of 0.5/NMS threshold of
0.4 in the experimental runner. S3FD uses different preprocessing and
thresholds, so these parts must not be shared blindly.

## Robot acceptance checklist

Run the following checks on the actual robot:

- Confirm the SCRFD model loads without ONNX Runtime errors.
- Confirm CUDA is actually selected if the robot has a CUDA GPU; otherwise
  record CPU performance separately.
- Run at least 5–10 minutes of live video and audio.
- Confirm the processing rate stays above the camera rate, normally 25 FPS.
- Record average and 95th-percentile detector and total pipeline latency.
- Check RAM and GPU memory before and during the run.
- Check that face boxes remain stable and tracks do not continuously reset.
- Check that audio/video synchronization and speaking decisions remain valid.
- Compare the same recording with S3FD and SCRFD.
- Keep the S3FD path available until the SCRFD run is accepted.

The workstation result proves that the SCRFD implementation is functional and
faster on that GPU. It does not by itself prove the final robot performance.

The workstation production server was rebuilt and started successfully with
SCRFD on 2026-09-15. It loaded the SCRFD model and listened on ports 8895,
8896, 8897, and 8898. Set the robot app's server address to:

```text
192.168.0.44
```

Then press Start in the robot application. Keep the server terminal running
while testing.

## Reference results

Full SCRFD/S3FD benchmark:

```text
experiments/scrfd_cpp/RESULTS.md
```

New LR-ASD checkpoint evaluation:

```text
experiments/trained_model_test/RESULTS.md
```

Latest trained-model C++ artifacts:

```text
experiments/trained_model_test/results/20260912_003121/
```

Model checksums from this workspace:

```text
SCRFD-2.5G dynamic ONNX:
4dc5fa5cbab42768917f23464743b0825d0ea3eaf11cfe367f18cbb4721d18bf

LR-ASD epoch 32 checkpoint:
9f796480b4ad1b35af6b179b53ecdef90b1081c7a124feb7bdce8678b694ac57

Existing S3FD 270x480 checkpoint:
e86b6d4d350e77692cbc03efeadcf699919bfd8d26e15c94987eb4bbfee1f086
```
