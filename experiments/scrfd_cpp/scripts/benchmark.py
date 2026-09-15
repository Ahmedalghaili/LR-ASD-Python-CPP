#!/usr/bin/env python3
"""Run sequential, repeated comparisons and match detections by per-frame IoU."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
VIDEOS = {
    'wide': (PROJECT/'WhatsApp Video 2026-08-30 at 9.46.47 PM.mp4', 's3fd_206x466.pt'),
    'two_person': (PROJECT/'VID_20260824_145739.mp4', 's3fd_270x480.pt'),
}

def iou(a,b):
    inter=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
    union=(a['x2']-a['x1'])*(a['y2']-a['y1'])+(b['x2']-b['x1'])*(b['y2']-b['y1'])-inter
    return inter/union if union else 0

def read_rows(path):
    frames={}
    with path.open() as f:
        for row in csv.DictReader(f):
            row={k:float(v) for k,v in row.items()}
            frames.setdefault(int(row['frame']),[]).append(row)
    return frames

def compare(base_path,candidate_path):
    base, candidate=read_rows(base_path),read_rows(candidate_path)
    matched=agree=reference=candidates=switches=0
    mapping={}
    overlaps=[]
    for frame in sorted(set(base)|set(candidate)):
        aa,bb=base.get(frame,[]),candidate.get(frame,[])
        reference+=len(aa);candidates+=len(bb)
        pairs=sorted(((iou(a,b),i,j) for i,a in enumerate(aa) for j,b in enumerate(bb)),reverse=True)
        used_a,used_b=set(),set()
        for overlap,i,j in pairs:
            if overlap<.5 or i in used_a or j in used_b:continue
            used_a.add(i);used_b.add(j);matched+=1;overlaps.append(overlap)
            agree+=aa[i]['speaking']==bb[j]['speaking']
            track=aa[i]['track'];other=bb[j]['track']
            if track in mapping and mapping[track]!=other:switches+=1
            mapping[track]=other
    return dict(matched=matched,baseline_observations=reference,candidate_observations=candidates,
                unmatched_baseline=reference-matched,unmatched_candidate=candidates-matched,
                speaking_agreement_pct=100*agree/matched if matched else None,
                mean_matched_iou=statistics.mean(overlaps) if overlaps else None,
                candidate_track_changes_per_baseline_track=switches)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=640)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true',help='Resume completed runs in --output')
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    if args.resume and not args.output:parser.error('--resume requires --output')
    destination=args.output or ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=args.resume)
    config=vars(args).copy();config['output']=str(destination)
    config['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    config['videos']={name:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,(p,_) in VIDEOS.items()}
    config['models']={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PROJECT/'cpp/models/lr_asd.pt',*ROOT.glob('models/*.onnx'),*[(PROJECT/'cpp/models'/model) for _,model in VIDEOS.values()]]}
    config['source_sha256']={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/main.cpp',ROOT/'src/scrfd.hpp',PROJECT/'cpp/src/main.cpp']}
    records=[]
    if args.resume:
        previous=json.loads((destination/'config.json').read_text())
        for key in ['repeats','max_frames','width','height','gpu','videos','models','source_sha256']:
            if previous[key]!=config[key]:raise RuntimeError('Cannot resume: '+key+' changed')
        records=json.loads((destination/'runs.json').read_text())
    else:
        (destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for video_name,(video,s3fd) in VIDEOS.items():
        audio=destination/f'{video_name}.wav'
        audio_command=['ffmpeg','-v','error','-y','-i',str(video),'-ac','1','-ar','16000','-vn','-c:a','pcm_s16le']
        if args.max_frames:audio_command+=['-t',str(args.max_frames/25)]
        if not args.resume or not audio.exists():subprocess.run(audio_command+[str(audio)],check=True)
        variants=[('s3fd',PROJECT/'cpp/models'/s3fd),('scrfd_2.5g',ROOT/'models/scrfd_2.5g_bnkps.dynamic.onnx'),('scrfd_10g',ROOT/'models/scrfd_10g_bnkps.dynamic.onnx')]
        for repeat in range(args.repeats):
            order=variants[repeat%3:]+variants[:repeat%3]
            for label,model in order:
                output=destination/f'{video_name}_{label}_r{repeat+1}.mp4'
                if any(r['video']==video_name and r['variant']==label and r['repeat']==repeat+1 for r in records):
                    for artifact in [output,Path(str(output)+'.csv'),Path(str(output)+'.timing.csv')]:
                        if not artifact.exists():raise RuntimeError('Missing completed artifact: '+str(artifact))
                    print(f'Keeping completed {video_name} {label} repeat {repeat+1}',flush=True)
                    continue
                command=[str(ROOT/'run.sh'),'--detector','s3fd' if label=='s3fd' else 'scrfd','--model',str(model),
                         '--asd',str(PROJECT/'cpp/models/lr_asd.pt'),'--video',str(video),'--audio',str(audio),
                         '--output',str(output),'--width',str(args.width),'--height',str(args.height),
                         '--max-frames',str(args.max_frames),'--warmup','10','--device','cuda']
                print(f'Running {video_name} {label} repeat {repeat+1}/{args.repeats}',flush=True)
                with output.with_suffix('.log').open('w') as log:
                    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=log,text=True,check=True)
                metrics=json.loads(result.stdout.strip().splitlines()[-1])
                metrics.update(video=video_name,variant=label,repeat=repeat+1,command=command)
                records.append(metrics)
                (destination/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps({k:metrics[k] for k in ['pipeline_fps','detector_mean_ms','tracks_created']}),flush=True)
    summary=[]
    for video_name in VIDEOS:
        baseline=statistics.median(r['pipeline_fps'] for r in records if r['video']==video_name and r['variant']=='s3fd')
        for label in ['s3fd','scrfd_2.5g','scrfd_10g']:
            group=[r for r in records if r['video']==video_name and r['variant']==label]
            item=dict(video=video_name,variant=label)
            for key in ['pipeline_fps','detector_mean_ms','detector_p95_ms','frame_p95_ms','tracks_created','face_observations','asd_calls']:
                item[key]=statistics.median(r[key] for r in group)
            item['fps_min']=min(r['pipeline_fps'] for r in group);item['fps_max']=max(r['pipeline_fps'] for r in group)
            item['speedup']=item['pipeline_fps']/baseline
            if label!='s3fd':
                item['comparison']=compare(destination/f'{video_name}_s3fd_r1.mp4.csv',destination/f'{video_name}_{label}_r1.mp4.csv')
            summary.append(item)
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# SCRFD C++ benchmark','',f"GPU: {config['gpu']}. {args.repeats} runs per configuration; medians below.",
           f'SCRFD input: {args.width}×{args.height}; FP32 CUDA, score 0.5, NMS 0.4. S3FD uses the existing 0.25 scale, score 0.9 and NMS 0.1.','',
           '| Video | Detector | FPS | Range | Speedup | Detector ms | Tracks |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['video']} | {r['variant']} | {r['pipeline_fps']:.2f} | {r['fps_min']:.2f}–{r['fps_max']:.2f} | {r['speedup']:.2f}× | {r['detector_mean_ms']:.2f} | {r['tracks_created']} |")
    lines+=['','## Agreement with S3FD','', '| Video | Detector | Matched boxes | Unmatched S3FD / SCRFD | Speaking agreement | Mean IoU | Track changes |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if 'comparison' not in r:continue
        c=r['comparison'];agreement=f"{c['speaking_agreement_pct']:.2f}%" if c['speaking_agreement_pct'] is not None else 'N/A'
        overlap=f"{c['mean_matched_iou']:.3f}" if c['mean_matched_iou'] is not None else 'N/A'
        lines.append(f"| {r['video']} | {r['variant']} | {c['matched']} | {c['unmatched_baseline']} / {c['unmatched_candidate']} | {agreement} | {overlap} | {c['candidate_track_changes_per_baseline_track']} |")
    lines+=['','## Measurement scope','',
      '- Total FPS includes decoding, detector preprocessing/inference/decoding/NMS, transfers, tracking, cropping, LR-ASD, overlays, CSV writes and video encoding/finalization. It excludes model loading, MFCC preparation and ten warmup calls.',
      '- Detector latency includes preprocessing, GPU inference, output transfer and NMS; CUDA is synchronized. GPU runs are sequential. The pipeline processes 25 source-time frames per second without pacing.',
      '- Both variants use the original LR-ASD checkpoint, crops and tracking logic. Keypoint model outputs are unused. This compares deployable configurations and runtimes, not isolated architecture FLOPs.',
      '- Detections are greedily matched per frame at IoU ≥ 0.5; track IDs are not assumed to correspond. Speaking agreement uses matched observations, including initial default scores. Track changes count changes in the matched SCRFD ID for a baseline ID.',
      '- These videos have no manual face/speaking labels: agreement is not accuracy, and unmatched boxes are not proven misses or false positives. Face recall and ASD mAP need labeled evaluation.',
      '- Existing tracking can bridge gaps without restoring missing time steps; this inherited behavior may influence disagreement. Results apply to these two recordings and this hardware.',
      '- Historical FPS numbers used a different warmup/timing setup; use the freshly measured S3FD baseline here. Raw commands, hashes, timings, CSVs and annotated videos are retained beside this report.','']
    (destination/'REPORT.md').write_text('\n'.join(lines))
    print('Results:',destination,flush=True)

if __name__=='__main__':main()
