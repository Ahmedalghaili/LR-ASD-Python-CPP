# LR-ASD robot integration

`MainWindow` owns an `LRASDWorker`. The worker runs outside the GUI thread and
calls `mVABuffer.GetVideoAudioBuffer()` every 200 ms. It consumes only new
camera frames, takes the matching latest 16 kHz PCM audio window, performs
face detection/tracking, creates 112 x 112 grayscale face crops, extracts the
same 13-coefficient MFCC features as Python, and runs the official LR-ASD
weights.

## Realtime optimization and intentionally disabled work

The live path was changed as follows:

- Face detection runs once on the newest frame in each 200 ms snapshot. Older
  queued frames are not redetected.
- The newest face crop is repeated for the elapsed temporal slots so LR-ASD
  still receives its required 5--25 frame input shape. This reduces detection
  cost, but is less accurate during very fast face motion than offline
  per-frame detection.
- The previous hand-written O(N^2) MFCC Fourier loop is disabled and replaced
  with OpenCV's optimized DFT.
- The stale video buffer is reduced from 300 to 50 frames, and audio from
  160,000 to 32,000 samples.
- When optional overlays are disabled, the raw video preview transfers the
  decoded `cv::Mat` buffer instead of making two full-resolution pixel copies.

The following optional robot features remain disabled in
`json/ROGG16_SSWR.json` for the realtime test: facial-expression recognition,
pose estimation, hand landmarks, the separate InspireFace/dlib face pipeline,
visual compass, image saving, and server audio echo. The preview, Whisper, LR-
ASD, robot communication, and raw video display remain enabled. MediaPipe
graphs may still initialize at startup, but they do not process each frame
while their settings are false.

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
