# LR-ASD C++ implementation

This implementation uses LibTorch and OpenCV 4.11. It loads TorchScript exports
of the exact official AVA checkpoint and S3FD face detector.

Implemented pipeline:

- S3FD face detection (same weights, 0.25 scale, 0.9 threshold, 0.1 NMS)
- IoU face tracking matching the Python rolling real-time pipeline
- square padded 112x112 grayscale face crops
- PCM 16-bit, 16 kHz mono WAV input
- native 13-coefficient MFCC matching `python_speech_features` defaults
- four MFCC steps per 25 fps video frame
- rolling 25-frame LR-ASD inference every five frames
- annotated video and frame/track score CSV output

Both CPU and CUDA builds are available. The CUDA build links directly to the
CUDA-enabled LibTorch and CUDA 12.8 runtime bundled in `.venv`, so a system-wide
CUDA toolkit is not required.

## Build and run

```bash
.venv/bin/python cpp/export_torchscript.py
cmake -S cpp -B cpp/build_cpu \
  -DCMAKE_PREFIX_PATH="$PWD/cpp/third_party/libtorch;/usr/local/lib/cmake/opencv4" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build_cpu -j2

bash cpp/build_cuda.sh

cpp/build_cpu/lr_asd --video \
  cpp/models/lr_asd_cpu.pt cpp/models/s3fd_270x480.pt \
  baseline/inputs/person_test.mp4 \
  baseline/inputs/person_test/pyavi/audio.wav \
  baseline/cpp/video_only.mp4

LR_ASD_DEVICE=cuda LR_ASD_PACE=1 cpp/build_cuda/lr_asd --video \
  cpp/models/lr_asd.pt cpp/models/s3fd_206x466.pt \
  input.mp4 audio_16khz_mono.wav output.mp4
```

S3FD exports are shape-specific because TorchScript tracing fixes the detector
geometry. The exporter currently creates 480x270 and 466x206 variants, matching
the two tested videos at the official 0.25 scale.
