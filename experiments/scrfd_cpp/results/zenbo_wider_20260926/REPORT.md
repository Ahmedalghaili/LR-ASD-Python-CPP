# S3FD vs SCRFD-2.5G for the Zenbo Junior robot

Date: 2026-09-26

Both detectors are tested exactly as the robot server runs them:

- S3FD: `s3fd_270x480.onnx`, frame stretched to 480×270, score ≥ 0.9, NMS 0.1
  (`LRASDWorker.cpp`).
- SCRFD-2.5G: `scrfd_2.5g_bnkps.dynamic.onnx`, letterboxed to 480×288,
  score ≥ 0.5, NMS 0.4 (`SCRFDDetector.hpp`).

`detectors.py` is a line-by-line Python port of both, used for the accuracy
tests. `bench.cpp` includes the server's `SCRFDDetector.hpp` unchanged and a
verbatim copy of the S3FD `detect()` code for the speed test.

## 1. Accuracy on WIDER FACE validation (3,226 images, 39,708 labeled faces)

This is the standard face-detection benchmark. It was scored with the official
WIDER FACE evaluation code (`wider_eval_official.py` from Pytorch_Retinaface,
with a NumPy `bbox.py` in place of the Cython overlap function).

Official average precision (AP):

| Subset | S3FD (deployed) | SCRFD-2.5G (deployed) |
|---|---:|---:|
| Easy | 80.3% | **90.9%** |
| Medium | 69.7% | **86.2%** |
| Hard | 34.5% | **56.2%** |

At the thresholds the robot actually uses (`op_point.py`):

| Subset | S3FD recall | SCRFD recall | S3FD precision | SCRFD precision |
|---|---:|---:|---:|---:|
| Easy | 65.5% | **85.0%** | **99.4%** | 94.5% |
| Medium | 49.2% | **75.6%** | **99.6%** | 96.6% |
| Hard | 21.0% | **38.9%** | **99.6%** | 97.2% |

Recall by face height (all faces), deployed thresholds:

| Face height | S3FD | SCRFD |
|---|---:|---:|
| 16–32 px | 2.1% | **23.4%** |
| 32–64 px | 37.1% | **69.2%** |
| 64–128 px | 65.7% | **85.7%** |
| ≥ 128 px | 82.6% | **93.3%** |

SCRFD finds far more faces at every size. S3FD produces slightly fewer false
detections because its 0.9 threshold is very strict, but it misses many more
real faces.

Both scores are lower than the published full-resolution results because the
robot runs both models on small inputs (about 480 px wide) for speed.

## 2. Zenbo Junior camera frames (466 frames, 640×480)

The frames are raw camera images saved by the robot server on 2026-09-17.

| | S3FD | SCRFD-2.5G |
|---|---:|---:|
| Frames with a detected face | 389 | **402** |
| Faces detected | 392 | **408** |

The detectors disagreed on 18 frames. Every one was inspected by eye
(`zenbo_disagreements_*.jpg`):

- SCRFD found a real face that S3FD missed in 17 frames: faces partly covered
  by a hand, turned sideways, looking down, cut off at the frame edge, or small
  on a monitor.
- S3FD found a real face that SCRFD missed in 1 frame (a small face on a
  monitor).
- Neither detector produced a false detection in these frames.

The 64 frames where both found nothing were also inspected
(`zenbo_no_detection_*.jpg`). About 50 show no face. About 12 show a heavily
hidden face (hand over the face or head bowed), and both detectors missed
these same frames.

## 3. Speed on Zenbo frames (GPU)

RTX 4080 SUPER, CUDA, 640×480 Zenbo frames. Each timing includes the full
`detect()` call: preprocessing, inference, decoding and NMS. The two
detectors ran in alternating blocks of 50 frames, 40 rounds, 2,000 calls
each.

**The GPU was shared with an unrelated training job running at 99–100%
utilization.** The absolute times are therefore higher than on an idle GPU.
The comparison is still fair because both detectors ran under the same load.

| | S3FD | SCRFD-2.5G |
|---|---:|---:|
| Mean | 11.42 ms | **3.20 ms** |
| Median | 12.65 ms | **3.09 ms** |
| 95th percentile | 13.27 ms | **5.09 ms** |
| 99th percentile | 14.55 ms | **5.45 ms** |

SCRFD was faster in 40 of 40 rounds, by 4.02× to 4.13× (median 4.09×).
With an idle GPU, the earlier benchmark measured about 2× (4–5 ms vs about
2.2 ms). Under load, S3FD slows down more because it is the heavier model.

## Conclusion

For the Zenbo Junior, SCRFD-2.5G is better than S3FD. As deployed, it:

- finds more faces: WIDER FACE AP +10.6 points on Easy, +16.5 on Medium and
  +21.7 on Hard, and 17 extra real faces vs 1 on Zenbo frames;
- runs 2–4× faster;
- is 27× smaller (3.3 MB vs 89.9 MB).

S3FD's only advantage is slightly fewer false detections (99.6% vs 97.2%
precision on WIDER hard). On the Zenbo frames neither model produced any false
detection.

Still to do: repeat the speed test on an idle GPU to get final absolute
numbers, and run a live end-to-end test with the Zenbo robot.

## Reproduce

```bash
# WIDER FACE predictions (needs WIDER_val + ground-truth .mat files in ./wider)
python wider_run.py s3fd; python wider_run.py scrfd
python wider_eval_official.py -p wider_pred_s3fd/ -g wider/
python wider_eval_official.py -p wider_pred_scrfd/ -g wider/
python op_point.py
# Zenbo frames
python zenbo_run.py; python zenbo_sheet.py
# Speed (compile bench.cpp against the server's ONNX Runtime GPU and OpenCV)
./bench ~/Downloads/raw_images s3fd_270x480.onnx scrfd_2.5g_bnkps.dynamic.onnx times.csv 40
```

The raw predictions (`wider_predictions.tar.gz`, 25 MB) and the evidence
images (`zenbo_*.jpg`) are kept on the workstation only and are not in the
public repository, because the images show a person's face. Both can be
regenerated with the commands above.
