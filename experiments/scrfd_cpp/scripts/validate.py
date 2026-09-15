#!/usr/bin/env python3
"""Compare experimental C++ boxes with InsightFace's pinned SCRFD reference."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from benchmark import ROOT, VIDEOS, read_rows, iou

parser=argparse.ArgumentParser()
parser.add_argument('results',type=Path)
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('scrfd_reference', ROOT/'third_party/scrfd_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
config=json.loads((args.results/'config.json').read_text())
width,height=config['width'],config['height']
checks=[]
for video,(path,_) in VIDEOS.items():
    for variant in ['scrfd_2.5g','scrfd_10g']:
        rows=read_rows(args.results/f'{video}_{variant}_r1.mp4.csv')
        session=ort.InferenceSession(str(ROOT/f'models/{variant}_bnkps.dynamic.onnx'),providers=['CPUExecutionProvider'])
        detector=reference.SCRFD(session=session)
        detector.prepare(-1,input_size=(width,height))
        cap=cv2.VideoCapture(str(path));sfps=cap.get(cv2.CAP_PROP_FPS)
        runs=json.loads((args.results/'runs.json').read_text())
        total=next(r['frames'] for r in runs if r['video']==video and r['variant']==variant and r['repeat']==1)
        sampled=sorted(set(np.linspace(0,total-1,min(total,8),dtype=int).tolist()))
        worst=0.;observations=0;source_index=-1
        for frame_index in sampled:
            want=math.floor(frame_index*sfps/25+.5)
            while source_index<want:
                ok,frame=cap.read();assert ok
                source_index+=1
            boxes,_=detector.detect(frame,thresh=.5,input_size=(width,height))
            actual=rows.get(frame_index,[])
            assert len(boxes)==len(actual),(video,variant,frame_index,len(boxes),len(actual))
            unmatched=set(range(len(actual)))
            for box in boxes:
                expected=dict(zip(['x1','y1','x2','y2'],box[:4]))
                j=max(unmatched,key=lambda index:iou(expected,actual[index]));unmatched.remove(j)
                error=max(abs(float(box[k])-actual[j][key]) for k,key in enumerate(['x1','y1','x2','y2']))
                worst=max(worst,error);observations+=1
                assert error<.1,(video,variant,frame_index,error)
        cap.release()
        checks.append(dict(video=video,variant=variant,frames=len(sampled),boxes=observations,max_coordinate_error_px=worst))
print(json.dumps(checks,indent=2))
(args.results/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
