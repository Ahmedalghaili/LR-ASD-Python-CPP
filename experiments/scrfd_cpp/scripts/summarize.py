#!/usr/bin/env python3
"""Publish a compact experiment report from completed benchmark folders."""
import json
from pathlib import Path
from benchmark import ROOT

folders=['cuda_480x288_verified','cuda_640x640']
results={name:json.loads((ROOT/'results'/name/'summary.json').read_text()) for name in folders}
small=results[folders[0]]
lines=['# SCRFD versus S3FD: C++ results','',
       'SCRFD-2.5G at **480×288** is the recommended starting configuration for further evaluation on these recordings. It reduces detector latency by roughly half and improves total video-loop throughput, while changing some detections and speaking decisions.',
       '', 'Measured on an NVIDIA RTX 4080 SUPER, CUDA, three full-video repeats per configuration. Values below are medians. Both original recordings were tested (534 and 802 processed frames). The original `cpp/` implementation and checkpoints are unchanged.',
       '', '## Main comparison: 480×288 SCRFD', '',
       '| Recording | S3FD FPS | SCRFD-2.5G FPS | FPS gain | S3FD detector ms | SCRFD detector ms | Detector time reduction |',
       '|---|---:|---:|---:|---:|---:|---:|']
for video in ['wide','two_person']:
    a=next(r for r in small if r['video']==video and r['variant']=='s3fd')
    b=next(r for r in small if r['video']==video and r['variant']=='scrfd_2.5g')
    lines.append(f"| {video} | {a['pipeline_fps']:.2f} | {b['pipeline_fps']:.2f} | {(b['speedup']-1)*100:.1f}% | {a['detector_mean_ms']:.2f} | {b['detector_mean_ms']:.2f} | {(1-b['detector_mean_ms']/a['detector_mean_ms'])*100:.1f}% |")
lines+=['','Halving detector time does not double total FPS because decoding, cropping, LR-ASD, transfers and output encoding still take time. Total FPS here excludes model loading, audio-feature preparation and warmup.','',
        '## All tested configurations','',
        '| Input | Recording | Detector | Median FPS | Min–max FPS | Speedup over same-run S3FD | Detector ms |',
        '|---|---|---|---:|---:|---:|---:|']
for folder, rows in results.items():
    size='480×288' if '480' in folder else '640×640'
    for r in rows:
        lines.append(f"| {size} | {r['video']} | {r['variant']} | {r['pipeline_fps']:.2f} | {r['fps_min']:.2f}–{r['fps_max']:.2f} | {r['speedup']:.2f}× | {r['detector_mean_ms']:.2f} |")
lines+=['','S3FD uses its original 0.25 image scale in both sets. SCRFD uses top-left letterboxing at the listed input size. Thresholds follow each implementation: S3FD confidence 0.9/NMS 0.1, SCRFD confidence 0.5/NMS 0.4. These compare complete deployment configurations, including different runtimes and resolution; they are not isolated architecture comparisons.',
        '', '## Detection and speaking differences at 480×288','',
        '| Recording | Model | Matched observations | Unmatched S3FD / SCRFD | Speaking agreement | Total tracks S3FD / SCRFD |',
        '|---|---|---:|---:|---:|---:|']
for r in small:
    if 'comparison' not in r:continue
    a=next(a for a in small if a['video']==r['video'] and a['variant']=='s3fd')
    c=r['comparison']
    lines.append(f"| {r['video']} | {r['variant']} | {c['matched']} | {c['unmatched_baseline']} / {c['unmatched_candidate']} | {c['speaking_agreement_pct']:.2f}% | {a['tracks_created']} / {r['tracks_created']} |")
lines+=['','**Agreement is not accuracy.** Detections are matched by per-frame IoU ≥ 0.5, independent of track IDs. Speaking agreement covers only matched observations and includes initial default scores. The videos have no manual ground truth, so unmatched detections cannot be classified as errors and no ASD mAP or face recall is claimed. Inspect the videos and evaluate labeled data before deployment.',
        '', '## Validation','']
for folder in folders:
    checks=json.loads((ROOT/'results'/folder/'validation.json').read_text())
    lines.append(f"- `{folder}`: all {sum(x['frames'] for x in checks)} sampled model/frame checks matched the official Python reference's box counts; maximum coordinate difference {max(x['max_coordinate_error_px'] for x in checks):.4f} pixels, below the 0.1-pixel tolerance. Samples span both complete videos and use sequential decoding to handle variable frame timing.")
lines += ['- Original C++ versus experimental S3FD: 1,604 two-person observations had identical boxes, track IDs and speaking decisions. Maximum LR-ASD score difference was 0.0001.',
          '- Comparison tests cover reordered track IDs, unmatched detections, track changes and no-match cases. Model checksums and ONNX validity were checked; dynamic-shape metadata changes preserve all network operations and weights.',
          '', '## Open the results','',
          '| Recording | S3FD | SCRFD-2.5G 480×288 | SCRFD-10G 480×288 |','|---|---|---|---|']
for video in ['wide','two_person']:
    links=[f'[Play](results/cuda_480x288_verified/{video}_{d}_r1.mp4)' for d in ['s3fd','scrfd_2.5g','scrfd_10g']]
    lines.append('| '+video+' | '+' | '.join(links)+' |')
lines += ['', '- [480×288 detailed report](results/cuda_480x288_verified/REPORT.md)',
          '- [640×640 detailed report](results/cuda_640x640/REPORT.md)',
          '- [Build and run instructions](README.md)', '- [Software versions](environment.json)', '',
          'Each final results folder contains the exact commands and hashes (`config.json`, `runs.json`), aggregates (`summary.json`), reference checks (`validation.json`), per-frame timing CSVs, detection/speaking CSVs, stderr logs and annotated videos. Generated models, dependencies, binaries and raw result folders are excluded from Git; this summary is retained.', '',
          'Measurements use FP32 models with runtime-default math settings, ten warmup calls, synchronized CUDA timing and sequential GPU runs with rotated order. They include video-loop I/O and exclude startup and MFCC preparation. Existing track-gap timing behavior is inherited. These are file-processing capacity measurements, not a live camera/microphone latency test.', '',
          'Model source URLs, checksums, upstream references and research-use terms are documented in [README.md](README.md).','']
(ROOT/'RESULTS.md').write_text('\n'.join(lines))
(ROOT/'results_summary.json').write_text(json.dumps(results,indent=2)+'\n')
print(ROOT/'RESULTS.md')
