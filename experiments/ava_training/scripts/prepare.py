#!/usr/bin/env python3
"""Prepare existing AVA train/val downloads, with per-video resume markers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile

VERSION=1

def prepare_video(job):
    root,split,video,rows=job
    root=Path(root);cv2.setNumThreads(1)
    files=list((root/'orig_videos/trainval').glob(video+'.*'))
    files=[p for p in files if not p.name.endswith('.part')]
    if len(files)!=1:raise RuntimeError(f'Expected one original video for {video}')
    source=files[0]
    signature=dict(version=VERSION,size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,
                   rows_sha256=hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest())
    marker=root/'preparation'/split/(video+'.json');marker.parent.mkdir(parents=True,exist_ok=True)
    if marker.exists() and json.loads(marker.read_text()).get('signature')==signature:
        return dict(video=video,split=split,resumed=True)
    if shutil.disk_usage(root).free<20*1024**3:raise RuntimeError('Stopped to preserve 20 GiB free space')
    print(f'{split}/{video}: extracting audio and {len(rows)} face crops',flush=True)
    audio=root/'orig_audios/trainval'/(video+'.wav');audio.parent.mkdir(parents=True,exist_ok=True)
    # Each video is processed independently, keeping only its waveform in RAM.
    if not audio.exists():
        temporary=audio.with_suffix('.part.wav')
        subprocess.run(['ffmpeg','-v','error','-y','-threads','2','-i',str(source),'-ac','1','-vn','-c:a','pcm_s16le','-ar','16000','-threads','2',str(temporary)],check=True)
        temporary.replace(audio)
    sr,wave=wavfile.read(audio,mmap=True)
    if sr!=16000 or wave.ndim!=1:raise RuntimeError(f'Invalid audio: {audio}')
    count=0
    for entity,group in rows.groupby('entity_id',sort=False):
        group=group.sort_values('frame_timestamp')
        start=int(float(group.frame_timestamp.iloc[0])*sr);end=int(float(group.frame_timestamp.iloc[-1])*sr)
        if end<=start or end>len(wave):raise RuntimeError(f'Invalid audio interval: {entity}')
        target=root/'clips_audios'/split/video/(entity+'.wav');target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.part.wav');wavfile.write(temp,sr,wave[start:end]);temp.replace(target)
        count+=1
    del wave
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps<=0:raise RuntimeError(f'Cannot decode {source}')
    next_index=None;frame=None;last_target=None;written=0
    # The original OpenCV POS_MSEC seek rounds timestamp*fps to a frame index.
    # Read consecutive targets sequentially; seek only across larger gaps.
    for timestamp,group in rows.sort_values('frame_timestamp').groupby('frame_timestamp',sort=True):
        target_index=math.floor(float(timestamp)*fps+.5)
        if target_index!=last_target:
            if next_index is None or target_index<next_index or target_index-next_index>fps:
                if not cap.set(cv2.CAP_PROP_POS_MSEC,float(timestamp)*1000):
                    raise RuntimeError(f'Cannot seek {video} at {timestamp}')
                ok,frame=cap.read();next_index=target_index+1
            else:
                ok=True
                while next_index<=target_index:
                    ok,frame=cap.read();next_index+=1
                    if not ok:break
            if not ok or frame is None:raise RuntimeError(f'Decode failed: {video} at {timestamp}')
            last_target=target_index
        h,w=frame.shape[:2]
        for row in group.itertuples(index=False):
            x1=int(row.entity_box_x1*w);x2=int(row.entity_box_x2*w)
            y1=int(row.entity_box_y1*h);y2=int(row.entity_box_y2*h)
            if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise RuntimeError(f'Invalid crop for {row.entity_id}')
            target=root/'clips_videos'/split/video/row.entity_id/(f'{timestamp:.2f}.jpg')
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix('.part.jpg')
            if not cv2.imwrite(str(temp),frame[y1:y2,x1:x2]):raise RuntimeError(f'Cannot write {temp}')
            temp.replace(target);written+=1
        if written%1000<len(group) and shutil.disk_usage(root).free<20*1024**3:
            raise RuntimeError('Stopped to preserve 20 GiB free space')
    cap.release()
    result=dict(video=video,split=split,tracks=count,frames=written,signature=signature)
    temporary=marker.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(marker)
    return result

def validate(root):
    summary={}
    for split in ['train','val']:
        tracks=frames=0
        for line in (root/'csv'/f'{split}_loader.csv').read_text().splitlines():
            entity,length,*_=line.split('\t');length=int(length);video=entity[:11]
            audio=root/'clips_audios'/split/video/(entity+'.wav')
            folder=root/'clips_videos'/split/video/entity
            images=list(folder.glob('*.jpg'))
            if len(images)!=length:raise RuntimeError(f'{entity}: expected {length} crops, found {len(images)}')
            sr,w=wavfile.read(audio,mmap=True)
            if sr!=16000 or not len(w):raise RuntimeError(f'Invalid clip: {audio}')
            tracks+=1;frames+=length
        summary[split]=dict(tracks=tracks,frames=frames)
    (root/'preparation/READY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DATA READY: '+json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit-videos',type=int,default=0);args=p.parse_args()
    root=args.data.resolve();jobs=[]
    for split in ['train','val']:
        df=pd.read_csv(root/'csv'/f'{split}_orig.csv')
        for video,rows in df.groupby('video_id',sort=True):jobs.append((str(root),split,video,rows))
    if args.limit_videos:jobs=jobs[:args.limit_videos]
    print(f'Preparing {len(jobs)} videos with {args.workers} workers',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(prepare_video,job) for job in jobs]
        for index,f in enumerate(as_completed(futures),1):
            try:result=f.result()
            except BaseException:
                for pending in futures:pending.cancel()
                raise
            print(f'[{index}/{len(jobs)}] '+json.dumps(result),flush=True)
    if not args.limit_videos:validate(root)
