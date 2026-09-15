#!/usr/bin/env python3
"""Prepare the downloaded dataset, then automatically start the full training run."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[3]
RUN=PROJECT/'exps/ava_training_20260910'
DATA=PROJECT/'AVADataPath'
RUN.mkdir(parents=True,exist_ok=True)
lock=(RUN/'pipeline.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('The preparation/training pipeline is already running.')

def status(phase,**extra):
    payload=dict(phase=phase,updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),**extra)
    temporary=RUN/'status.tmp';temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(RUN/'status.json')
    print(json.dumps(payload),flush=True)

prepare=[sys.executable,str(PROJECT/'experiments/ava_training/scripts/prepare.py'),'--data',str(DATA),'--workers','4']
train=[sys.executable,'-u','train.py','--dataPathAVA',str(DATA),'--savePath',str(RUN),'--maxEpoch','40','--batchSize','1500','--nDataLoaderThread','8']
(RUN/'config.json').write_text(json.dumps(dict(dataset=str(DATA),prepare=prepare,train=train,initialization='random (from scratch)',detector='SCRFD remains an inference component; training uses AVA annotation crops'),indent=2)+'\n')
try:
    status('preparing')
    with (RUN/'preprocess.log').open('a') as log:
        subprocess.run(prepare,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (DATA/'preparation/READY.json').exists():raise RuntimeError('Dataset validation did not finish')
    status('training')
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (RUN/'train.log').open('a') as log:
        subprocess.run(train,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('complete')
except BaseException as error:
    status('failed',error=str(error))
    raise
