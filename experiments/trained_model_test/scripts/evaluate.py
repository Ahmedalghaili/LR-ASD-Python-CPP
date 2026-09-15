#!/usr/bin/env python3
"""Evaluate an explicit checkpoint on the complete, prepared AVA validation set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from ASD import ASD
from dataLoader import val_loader
from utils.get_ava_active_speaker_performance import run_evaluation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=PROJECT/'AVADataPath'
    dataset=val_loader(str(root/'csv/val_loader.csv'),str(root/'clips_audios/val'),str(root/'clips_videos/val'))
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
    net=ASD()
    net.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
    net.eval();scores=[];labels_all=[];start=time.monotonic()
    with torch.inference_mode():
        for i,(audio,visual,labels) in enumerate(loader,1):
            logits,_=net.model(audio[0].cuda(),visual[0].cuda())
            _,probabilities,_,_=net.lossAV(logits,labels[0].flatten().cuda())
            scores.extend(probabilities[:,1].cpu().numpy().tolist())
            labels_all.extend(labels[0].flatten().numpy().tolist())
            if i%500==0 or i==len(loader):print(f'{i}/{len(loader)} validation tracks',flush=True)
    df=pd.read_csv(root/'csv/val_orig.csv')
    if len(scores)!=len(df):raise RuntimeError(f'Prediction count {len(scores)} differs from annotation count {len(df)}')
    labels=np.asarray(labels_all);p=np.asarray(scores)
    if not np.isfinite(p).all():raise RuntimeError('Nonfinite predictions')
    if not np.array_equal(labels,(df.label_id.to_numpy()==1).astype(int)):
        raise RuntimeError('Loader label order does not match the annotation order')
    df['score']=p;df['label']='SPEAKING_AUDIBLE';df=df.drop(columns=['label_id','instance_id'])
    csv_path=args.output/'predictions.csv';df.to_csv(csv_path,index=False)
    with (root/'csv/val_orig.csv').open() as gt,csv_path.open() as pred:
        ap=float(run_evaluation(gt,pred))
    predicted=p>=.5
    tp=int(np.sum(predicted & (labels==1)));fp=int(np.sum(predicted & (labels==0)))
    fn=int(np.sum(~predicted & (labels==1)));tn=int(np.sum(~predicted & (labels==0)))
    result=dict(checkpoint=str(args.checkpoint.resolve()),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                split='AVA validation',tracks=len(dataset),frames=len(scores),AP_percent=ap,
                accuracy_percent=100*(tp+tn)/len(labels),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                F1=2*tp/max(1,2*tp+fp+fn),confusion=dict(TP=tp,FP=fp,FN=fn,TN=tn),
                threshold=.5,wall_seconds=time.monotonic()-start,torch_version=torch.__version__)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
