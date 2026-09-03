# LR-ASD robot integration

`MainWindow` owns an `LRASDWorker`. The worker runs outside the GUI thread and
calls `mVABuffer.GetVideoAudioBuffer()` every 40 ms. It consumes only new camera
frames, takes the matching latest 16 kHz PCM audio window, performs S3FD face
detection/tracking, creates 112 x 112 grayscale face crops, extracts the same
13-coefficient MFCC features as Python, and runs the official LR-ASD weights.

## Models

The large generated model files are intentionally not committed. From the
LR-ASD repository root, export them with:

```bash
python cpp/export_onnx.py
```

This creates `~/SoloSeniorWatchRobot_build/lr-asd/lr_asd.onnx` and
`s3fd_270x480.onnx`. Override that location with `LR_ASD_MODEL_DIR`.

The x86 robot build uses ONNX Runtime GPU 1.22 and the existing `USE_GPU`
option, so CUDA is selected when available. On ARM/aarch64 or when CUDA cannot
be initialized, ONNX Runtime runs the same graph on CPU.

On this workstation, launch the built robot application with the supplied
environment wrapper so ONNX Runtime can find the CUDA libraries bundled in the
Python environment:

```bash
bash run_with_lrasd.sh
```

## Runtime output

Each active track emits a JSON line containing `track`, `class1_logit`,
`speaking`, and `window_frames`. LR-ASD's official decision rule is class-1
logit >= 0. A performance record is emitted every 50 updates.

The queues have no timestamps. Synchronization therefore aligns the newest
audio sample to the newest received frame and assumes the robot sends video at
25 fps, as required by LR-ASD. If the robot sender uses another frame rate,
timestamps should be added to `VABuffer` and frames resampled to 25 fps.
